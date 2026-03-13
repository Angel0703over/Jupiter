"""
Communication handler for decoding tasks.

1. Last stage sends tree_candidates to first stage:
    1. last stage: tree_candidates_send_queues
    2. first stage: tree_candidates_receive_queues
2. Tree decoding computation:
    2.1 Send activation (all stages except the last)
    2.2 Receive activation (all stages except the first)
3. Last stage broadcasts new_token, others receive new_token.
"""

import threading
import time
import torch
import torch.distributed as dist

from . import threadsafe_queue,tag_manager
from .communication import recv_helper_thread,send_helper_thread

ROUND_END_POINT_ID = -1
STOP_SIGNAL = object()
THREAD_JOIN_TIMEOUT_SEC = 120.0

class CommunicationHandler():
    """Handles communication between stages."""
    def __init__(self, config):
        self.rank = config.stage
        self.world_size = config.total_stage
        self.next_rank = config.next_rank
        self.pre_rank = config.pre_rank
        self.if_first_rank = config.is_first_stage
        self.if_last_rank = config.is_last_stage

        # Since the number of new tokens is unknown, the first element is new_token_len.
        # select_indices is [:,1:new_token_len+1],
        # new_input_ids is [:, new_token_len+1:2*new_token_len+1].
        # The number of new tokens must be less than or equal to config.medusa_num_heads.

        # For reshape.
        self.tensor_shape = {
                            "tree_decoding": (1,  64, config.hidden_size),
                            "tree_candidates": (1,64),
                            "new_token":(1,1+2*config.medusa_num_heads)
                            }
        # For recv to allocate space.
        self.tensor_shape_for_recv = {
                            "tree_decoding": (1, int (1 + 64*config.hidden_size)),
                            "tree_candidates": (1, int (1+1*64)),
                            "new_token":(1, int (1+ 1+2*config.medusa_num_heads)  )
                            }
        self.tensor_type = {
                            "tree_decoding": config.torch_dtype,
                            "tree_candidates":torch.int64,
                            "new_token":torch.int64
                            } 
        self.tag_manager = tag_manager.Tag()
        self.tensor_tag = {
                            "tree_decoding":  self.tag_manager.get_next_tag(),  # Get incremental tag to keep different from other communication_handlers.
                            "tree_candidates":self.tag_manager.get_next_tag(),
                            "new_token": self.tag_manager.get_next_tag()
                            }
        
        self.device = config.device
        self.setup_queue()
        # Stop event to signal threads to stop.
        self.stop_event = threading.Event()
        # List to keep track of helper threads.
        self.helper_threads = []
        self.round_end_sent = False
        self.start_helper_threads()


    def setup_queue(self):
        """
        Set up queues for communication between main compute thread
        and helper communication threads. One queue per tensor
        in forward/backward direction.
        """
        # For tree candidates.
        if self.if_first_rank:
            self.tree_candidates_receive_queues = threadsafe_queue.Queue()
        if self.if_last_rank:
            self.tree_candidates_send_queues = threadsafe_queue.Queue()
        # For tree decoding.
        if not self.if_first_rank:
            self.tree_decoding_receive_queues = threadsafe_queue.Queue()
        if not self.if_last_rank:
            self.tree_decoding_send_queues = threadsafe_queue.Queue()
        # For new token.
        if self.if_last_rank:
            self.new_token_send_queues = threadsafe_queue.Queue()
        else:
            self.new_token_receive_queues = threadsafe_queue.Queue()
    def start_helper_threads(self):
        self.stop_event.clear()
        # For tree candidates
        if self.if_first_rank:
            self.start_helper_thread(
                func=tree_candidates_recv_helper_thread,
                args=(self.tree_candidates_receive_queues,
                    self.tensor_shape_for_recv["tree_candidates"],
                    self.world_size-1,
                    self.tensor_tag["tree_candidates"],
                    self.tensor_type["tree_candidates"],
                    self.stop_event),
                name="decode_tree_candidates_recv_helper"
            )

        if self.if_last_rank:
            self.start_helper_thread(
                func=send_helper_thread,
                args=(self.tree_candidates_send_queues,
                    0,
                    self.tensor_tag["tree_candidates"],
                    self.stop_event),
                name="decode_tree_candidates_send_helper"
            )

        # For tree decoding
        if not self.if_first_rank:
            self.start_helper_thread(
                func=recv_helper_thread,
                args=(self.tree_decoding_receive_queues,
                    self.tensor_shape_for_recv["tree_decoding"],
                    self.pre_rank,
                    self.tensor_tag["tree_decoding"],
                    self.tensor_type["tree_decoding"],
                    self.stop_event),
                name="decode_tree_decoding_recv_helper"
            )

        if not self.if_last_rank:
            self.start_helper_thread(
                func=send_helper_thread,
                args=(self.tree_decoding_send_queues,
                    self.next_rank,
                    self.tensor_tag["tree_decoding"],
                    self.stop_event),
                name="decode_tree_decoding_send_helper"
            )

        # For new token
        if self.if_last_rank:
            self.start_helper_thread(
                func=broadcast_send_helper_thread,
                args=(self.new_token_send_queues,
                    self.world_size-1,
                    self.stop_event),
                name="decode_new_token_broadcast_send_helper"
            )
        else:
            self.start_helper_thread(
                func=broadcast_recv_helper_thread,
                args=(self.new_token_receive_queues,
                    self.tensor_shape_for_recv["new_token"],
                    self.world_size-1,
                    self.tensor_type["new_token"],
                    self.stop_event),
                name="decode_new_token_broadcast_recv_helper"
            )


    def start_helper_thread(self, func, args, name=None):
        helper_thread = threading.Thread(target=func, args=args, daemon=True, name=name)
        helper_thread.start()
        self.helper_threads.append(helper_thread)  # Track the thread.

    def stop_helper_threads(self):
        print("[DecodingCommunicationHandler] stopping helper threads", flush=True)

        # Mark local threads as stopping first.
        self.stop_event.set()

        # Wake peer recv threads blocked in dist.recv / dist.broadcast directly.
        self._wake_peer_recv_threads_direct()

        # Wake local send threads blocked on queue.remove.
        if hasattr(self, "tree_candidates_send_queues"):
            self.tree_candidates_send_queues.add(STOP_SIGNAL)
        if hasattr(self, "tree_decoding_send_queues"):
            self.tree_decoding_send_queues.add(STOP_SIGNAL)
        if hasattr(self, "new_token_send_queues"):
            self.new_token_send_queues.add(STOP_SIGNAL)

        deadline = time.time() + THREAD_JOIN_TIMEOUT_SEC
        alive_threads = []
        for thread in self.helper_threads:
            while thread.is_alive() and time.time() < deadline:
                thread.join(timeout=0.05)
                time.sleep(0)
            if thread.is_alive():
                alive_threads.append(thread)

        self.helper_threads.clear()

        if alive_threads:
            tagged_alive_threads = []
            for thread in alive_threads:
                tag_name, tag_value = self._infer_tag_from_thread_name(thread.name or "")
                tagged_alive_threads.append(
                    f"{thread.name}(ident={thread.ident}, tag={tag_name}:{tag_value})"
                )
            detail = ", ".join(tagged_alive_threads)
            print(
                "[DecodingCommunicationHandler] still alive after stop: "
                f"{detail}; tensor_tag_map={self.tensor_tag}",
                flush=True,
            )
            raise RuntimeError(
                "[DecodingCommunicationHandler] helper thread recycle failed: "
                f"{detail}; tensor_tag_map={self.tensor_tag}"
            )
        print("[DecodingCommunicationHandler] stopped", flush=True)

    def _infer_tag_from_thread_name(self, thread_name):
        if "tree_candidates" in thread_name:
            return "tree_candidates", self.tensor_tag["tree_candidates"]
        if "tree_decoding" in thread_name:
            return "tree_decoding", self.tensor_tag["tree_decoding"]
        if "new_token" in thread_name:
            return "new_token", self.tensor_tag["new_token"]
        return "unknown", "unknown"

    def _wake_peer_recv_threads_direct(self):
        # Wake first-rank tree_candidates recv helper.
        if self.if_last_rank and not self.round_end_sent:
            try:
                self._direct_send(self._make_stop_payload("tree_candidates"), 0, self.tensor_tag["tree_candidates"])
            except RuntimeError as exc:
                print(f"[DecodingCommunicationHandler] direct stop send tree_candidates failed: {exc}", flush=True)

        # Wake non-first-rank tree_decoding recv helper.
        if not self.if_last_rank:
            try:
                self._direct_send(self._make_stop_payload("tree_decoding"), self.next_rank, self.tensor_tag["tree_decoding"])
            except RuntimeError as exc:
                print(f"[DecodingCommunicationHandler] direct stop send tree_decoding failed: {exc}", flush=True)

        # Wake new_token broadcast recv helper on non-last ranks.
        if self.if_last_rank and not self.round_end_sent:
            try:
                _broadcast_send(self._make_round_end_payload(), self.world_size - 1)
            except RuntimeError as exc:
                print(f"[DecodingCommunicationHandler] direct stop broadcast failed: {exc}", flush=True)

    def _direct_send(self, tensor, dst_rank, tag):
        if tensor.device != torch.device("cpu"):
            tensor = tensor.cpu()
        dist.send(tensor=tensor, dst=dst_rank, tag=tag)

    def _make_stop_payload(self, tensor_name):
        tensor = torch.zeros(
            self.tensor_shape_for_recv[tensor_name],
            dtype=self.tensor_type[tensor_name],
        )
        if tensor_name == "tree_candidates":
            tensor.view(-1)[0] = ROUND_END_POINT_ID
        return tensor

    def _make_round_end_payload(self):
        tensor = torch.zeros(
            self.tensor_shape_for_recv["new_token"],
            dtype=self.tensor_type["new_token"],
        )
        tensor.view(-1)[0] = ROUND_END_POINT_ID
        return tensor
        
    def flatten_before_send(self,tensor, point_id):
        flattened_tensor = tensor.reshape(1,-1)
        point_id_tensor = torch.tensor([point_id], dtype=flattened_tensor.dtype, device=flattened_tensor.device)
        point_id_tensor = point_id_tensor.reshape(1,-1)
        result_tensor = torch.cat((point_id_tensor, flattened_tensor),dim=1)
        return result_tensor
    def reshape_after_recv(self,tensor,tag):
        if tag == self.tensor_tag["tree_decoding"]:
            shape = self.tensor_shape["tree_decoding"]
        elif tag == self.tensor_tag["tree_candidates"]:
            shape = self.tensor_shape["tree_candidates"]
        elif tag == self.tensor_tag["new_token"]:
            shape =  self.tensor_shape["new_token"]
        point_id = int(tensor[0][0].item())
        reshaped_tensor = tensor[:, 1:].reshape(shape)
        return reshaped_tensor,point_id
    def send(self, tensor, tag, point_id): 
        tensor = self.flatten_before_send(tensor, point_id)
        if tag == self.tensor_tag["tree_decoding"]:
            self.tree_decoding_send_queues.add(tensor)
        elif tag == self.tensor_tag["tree_candidates"]:
            self.tree_candidates_send_queues.add(tensor)
        elif tag == self.tensor_tag["new_token"]:
            if point_id == ROUND_END_POINT_ID:
                self.round_end_sent = True
            self.new_token_send_queues.add(tensor)
        else:
            raise NotImplementedError
    def recv(self, tag):
        if tag == self.tensor_tag["tree_decoding"]:
            tensor =  self.tree_decoding_receive_queues.remove()
            # tensor = tensor.requires_grad_()
        elif tag == self.tensor_tag["tree_candidates"]:
            tensor =  self.tree_candidates_receive_queues.remove()
        elif tag == self.tensor_tag["new_token"]:
            tensor =  self.new_token_receive_queues.remove()
        else:
            raise NotImplementedError
        if self.device == "cuda":
            tensor = tensor.cuda()
        tensor, point_id = self.reshape_after_recv(tensor,tag)
        return tensor, point_id
    

def broadcast_send_helper_thread(send_queue, src,stop_event):
    thread_name = threading.current_thread().name
    print(f"[broadcast_send_helper_thread:{thread_name}] started", flush=True)
    while True:
        tensor = send_queue.remove()
        if tensor is STOP_SIGNAL:
            print(f"[broadcast_send_helper_thread:{thread_name}] thread exit", flush=True)
            break
        _broadcast_send(tensor, src)

def broadcast_recv_helper_thread(recv_queue, tensor_shape, src_rank, dtype, stop_event):
    thread_name = threading.current_thread().name
    print(
        f"[broadcast_recv_helper_thread:{thread_name}] started (src={src_rank})",
        flush=True,
    )

    while True:
        tensor = _broadcast_recv(tensor_shape, src_rank, dtype)
        point_id = int(tensor.view(-1)[0].item())
        if point_id == ROUND_END_POINT_ID:
            print(
                f"[broadcast_recv_helper_thread:{thread_name}] round end received",
                flush=True,
            )
            recv_queue.add(tensor)
            break
        recv_queue.add(tensor)
    print(f"[broadcast_recv_helper_thread:{thread_name}] thread exit", flush=True)


def tree_candidates_recv_helper_thread(
    recv_queue,
    tensor_shape,
    src_rank,
    tag,
    dtype,
    stop_event,
):
    thread_name = threading.current_thread().name
    print(
        f"[recv_helper_thread:{thread_name}] started (tag={tag}, src={src_rank})",
        flush=True,
    )
    while True:
        tensor = torch.zeros(tensor_shape, dtype=dtype)
        dist.recv(tensor=tensor, src=src_rank, tag=tag)
        point_id = int(tensor.view(-1)[0].item())
        recv_queue.add(tensor)
        if point_id == ROUND_END_POINT_ID:
            print(
                f"[recv_helper_thread:{thread_name}] round end received",
                flush=True,
            )
            break
    print(f"[recv_helper_thread:{thread_name}] thread exit", flush=True)

def _broadcast_send(tensor, src_rank ):
    if tensor.device != torch.device("cpu"): # for gloo 
        tensor = tensor.cpu()
    dist.broadcast(tensor=tensor , src=src_rank)


def _broadcast_recv(tensor_shape, src_rank,dtype):
    tensor = torch.zeros(tensor_shape, dtype=dtype) 
    dist.broadcast(tensor, src=src_rank)
    return tensor
    
