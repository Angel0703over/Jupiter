"""
Communication handler for decoding tasks.
"""

import threading
import torch
import torch.distributed as dist

from . import threadsafe_queue, tag_manager
from .communication import recv_helper_thread, send_helper_thread

ROUND_END_POINT_ID = -1
STOP_SIGNAL = object()   # 用于唤醒 send 线程


class CommunicationHandler():
    def __init__(self, config):
        print(f"DecodingCommunicationHandler.__init__ id={id(self)}", flush=True)

        self.rank = config.stage
        self.world_size = config.total_stage
        self.next_rank = config.next_rank
        self.pre_rank = config.pre_rank
        self.if_first_rank = config.is_first_stage
        self.if_last_rank = config.is_last_stage

        self.tensor_shape = {
            "tree_decoding": (1, 64, config.hidden_size),
            "tree_candidates": (1, 64),
            "new_token": (1, 1 + 2 * config.medusa_num_heads)
        }

        self.tensor_shape_for_recv = {
            "tree_decoding": (1, int(1 + 64 * config.hidden_size)),
            "tree_candidates": (1, int(1 + 64)),
            "new_token": (1, int(1 + 1 + 2 * config.medusa_num_heads))
        }

        self.tensor_type = {
            "tree_decoding": config.torch_dtype,
            "tree_candidates": torch.int64,
            "new_token": torch.int64
        }

        self.tag_manager = tag_manager.Tag()

        self.tensor_tag = {
            "tree_decoding": self.tag_manager.get_next_tag(),
            "tree_candidates": self.tag_manager.get_next_tag(),
            "new_token": self.tag_manager.get_next_tag()
        }

        self.device = config.device

        self.setup_queue()

        self.stop_event = threading.Event()
        self.helper_threads = []

        self.start_helper_threads()

    def setup_queue(self):

        if self.if_first_rank:
            self.tree_candidates_receive_queues = threadsafe_queue.Queue()

        if self.if_last_rank:
            self.tree_candidates_send_queues = threadsafe_queue.Queue()

        if not self.if_first_rank:
            self.tree_decoding_receive_queues = threadsafe_queue.Queue()

        if not self.if_last_rank:
            self.tree_decoding_send_queues = threadsafe_queue.Queue()

        if self.if_last_rank:
            self.new_token_send_queues = threadsafe_queue.Queue()
        else:
            self.new_token_receive_queues = threadsafe_queue.Queue()

    def start_helper_threads(self):

        print(f"[rank {self.rank}] start_helper_threads handler_id={id(self)} existing_threads={len(self.helper_threads)}", flush=True)

        self.stop_event.clear()

        # tree_candidates
        if self.if_first_rank:
            self.start_helper_thread(
                recv_helper_thread,
                (
                    self.tree_candidates_receive_queues,
                    self.tensor_shape_for_recv["tree_candidates"],
                    self.world_size - 1,
                    self.tensor_tag["tree_candidates"],
                    self.tensor_type["tree_candidates"],
                    self.stop_event
                )
            )

        if self.if_last_rank:
            self.start_helper_thread(
                send_helper_thread,
                (
                    self.tree_candidates_send_queues,
                    0,
                    self.tensor_tag["tree_candidates"],
                    self.stop_event
                )
            )

        # tree_decoding
        if not self.if_first_rank:
            self.start_helper_thread(
                recv_helper_thread,
                (
                    self.tree_decoding_receive_queues,
                    self.tensor_shape_for_recv["tree_decoding"],
                    self.pre_rank,
                    self.tensor_tag["tree_decoding"],
                    self.tensor_type["tree_decoding"],
                    self.stop_event
                )
            )

        if not self.if_last_rank:
            self.start_helper_thread(
                send_helper_thread,
                (
                    self.tree_decoding_send_queues,
                    self.next_rank,
                    self.tensor_tag["tree_decoding"],
                    self.stop_event
                )
            )

        # new_token broadcast
        if self.if_last_rank:

            self.start_helper_thread(
                broadcast_send_helper_thread,
                (
                    self.new_token_send_queues,
                    self.world_size - 1,
                    self.stop_event
                )
            )

        else:

            self.start_helper_thread(
                broadcast_recv_helper_thread,
                (
                    self.new_token_receive_queues,
                    self.tensor_shape_for_recv["new_token"],
                    self.world_size - 1,
                    self.tensor_type["new_token"],
                    self.stop_event
                )
            )

        print(f"[rank {self.rank}] start_helper_threads handler_id={id(self)} existing_threads={len(self.helper_threads)}", flush=True)

    def start_helper_thread(self, func, args):
        helper_thread = threading.Thread(target=func, args=args, daemon=True)
        helper_thread.start()
        self.helper_threads.append(helper_thread)

    def stop_helper_threads(self):
        print(f"[rank {self.rank}] stop_helper_threads handler_id={id(self)} helper_threads={len(self.helper_threads)}", flush=True)
        # First enqueue wake-up payloads so peer recv/broadcast threads can return
        # from blocking collectives and observe stop_event.
        if hasattr(self, "tree_candidates_send_queues"):
            self.tree_candidates_send_queues.add(self._make_stop_payload("tree_candidates"))
        if hasattr(self, "tree_decoding_send_queues"):
            self.tree_decoding_send_queues.add(self._make_stop_payload("tree_decoding"))
        if self.if_last_rank and hasattr(self, "new_token_send_queues"):
            self.new_token_send_queues.add(self._make_round_end_payload())

        self.stop_event.set()

        # Wake local send threads blocked on queue.remove().
        if self.if_last_rank:
            self.new_token_send_queues.add(STOP_SIGNAL)
        if hasattr(self, "tree_candidates_send_queues"):
            self.tree_candidates_send_queues.add(STOP_SIGNAL)
        if hasattr(self, "tree_decoding_send_queues"):
            self.tree_decoding_send_queues.add(STOP_SIGNAL)
        for t in self.helper_threads:
            print(f"[rank {self.rank}] joining {t}", flush=True)
            if t.is_alive():
                t.join(timeout=1.0)
            print(f"[rank {self.rank}] alive_after_join={t.is_alive()} thread={t}", flush=True)
        self.helper_threads.clear()
        print(f"[rank {self.rank}] stop_helper_threads finished", flush=True)

    def _make_stop_payload(self, tensor_name):
        dtype = self.tensor_type[tensor_name]
        shape = self.tensor_shape_for_recv[tensor_name]
        return torch.zeros(shape, dtype=dtype)

    def _make_round_end_payload(self):
        shape = self.tensor_shape_for_recv["new_token"]
        tensor = torch.zeros(shape, dtype=self.tensor_type["new_token"])
        tensor.view(-1)[0] = ROUND_END_POINT_ID
        return tensor

    def flatten_before_send(self, tensor, point_id):

        flattened_tensor = tensor.reshape(1, -1)

        point_id_tensor = torch.tensor(
            [point_id],
            dtype=flattened_tensor.dtype,
            device=flattened_tensor.device
        ).reshape(1, -1)

        return torch.cat((point_id_tensor, flattened_tensor), dim=1)

    def reshape_after_recv(self, tensor, tag):

        if tag == self.tensor_tag["tree_decoding"]:
            shape = self.tensor_shape["tree_decoding"]

        elif tag == self.tensor_tag["tree_candidates"]:
            shape = self.tensor_shape["tree_candidates"]

        elif tag == self.tensor_tag["new_token"]:
            shape = self.tensor_shape["new_token"]

        point_id = int(tensor[0][0].item())

        reshaped_tensor = tensor[:, 1:].reshape(shape)

        return reshaped_tensor, point_id

    def send(self, tensor, tag, point_id):

        tensor = self.flatten_before_send(tensor, point_id)

        if tag == self.tensor_tag["tree_decoding"]:
            self.tree_decoding_send_queues.add(tensor)

        elif tag == self.tensor_tag["tree_candidates"]:
            self.tree_candidates_send_queues.add(tensor)

        elif tag == self.tensor_tag["new_token"]:
            self.new_token_send_queues.add(tensor)

        else:
            raise NotImplementedError

    def recv(self, tag):

        if tag == self.tensor_tag["tree_decoding"]:
            tensor = self.tree_decoding_receive_queues.remove()

        elif tag == self.tensor_tag["tree_candidates"]:
            tensor = self.tree_candidates_receive_queues.remove()

        elif tag == self.tensor_tag["new_token"]:
            tensor = self.new_token_receive_queues.remove()

        else:
            raise NotImplementedError

        if self.device == "cuda":
            tensor = tensor.cuda()

        tensor, point_id = self.reshape_after_recv(tensor, tag)

        return tensor, point_id


def broadcast_send_helper_thread(send_queue, src, stop_event):

    while True:

        tensor = send_queue.remove()

        if tensor is STOP_SIGNAL:
            print("[broadcast_send_helper_thread] stop signal received", flush=True)
            break

        _broadcast_send(tensor, src)

        if stop_event.is_set():
            break


def broadcast_recv_helper_thread(recv_queue, tensor_shape, src_rank, dtype, stop_event):

    print("[broadcast_recv_helper_thread] started", flush=True)

    while True:

        tensor = _broadcast_recv(tensor_shape, src_rank, dtype)

        point_id = int(tensor.view(-1)[0].item())

        if point_id == ROUND_END_POINT_ID:
            print("[broadcast_recv_helper_thread] round end received -> exit", flush=True)
            break

        recv_queue.add(tensor)

        if stop_event.is_set():
            break


def _broadcast_send(tensor, src_rank):

    if tensor.device != torch.device("cpu"):
        tensor = tensor.cpu()

    dist.broadcast(tensor=tensor, src=src_rank)


def _broadcast_recv(tensor_shape, src_rank, dtype):

    tensor = torch.zeros(tensor_shape, dtype=dtype)

    dist.broadcast(tensor, src=src_rank)

    return tensor
