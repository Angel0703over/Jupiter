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
import torch
import torch.distributed as dist

from . import threadsafe_queue,tag_manager
from .communication import recv_helper_thread,send_helper_thread

ROUND_END_POINT_ID = -1
STOP_SIGNAL = object()
THREAD_JOIN_TIMEOUT_SEC = 20.0

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
                func=recv_helper_thread,
                args=(self.tree_candidates_receive_queues,
                    self.tensor_shape_for_recv["tree_candidates"],
                    self.world_size-1,
                    self.tensor_tag["tree_candidates"],
                    self.tensor_type["tree_candidates"],
                    self.stop_event)
            )

        if self.if_last_rank:
            self.start_helper_thread(
                func=send_helper_thread,
                args=(self.tree_candidates_send_queues,
                    0,
                    self.tensor_tag["tree_candidates"],
                    self.stop_event)
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
                    self.stop_event)
            )

        if not self.if_last_rank:
            self.start_helper_thread(
                func=send_helper_thread,
                args=(self.tree_decoding_send_queues,
                    self.next_rank,
                    self.tensor_tag["tree_decoding"],
                    self.stop_event)
            )

        # For new token
        if self.if_last_rank:
            self.start_helper_thread(
                func=broadcast_send_helper_thread,
                args=(self.new_token_send_queues,
                    self.world_size-1,
                    self.stop_event)
            )
        else:
            self.start_helper_thread(
                func=broadcast_recv_helper_thread,
                args=(self.new_token_receive_queues,
                    self.tensor_shape_for_recv["new_token"],
                    self.world_size-1,
                    self.tensor_type["new_token"],
                    self.stop_event)
            )


    def start_helper_thread(self, func, args):
        helper_thread = threading.Thread(target=func, args=args, daemon=True)
        helper_thread.start()
        self.helper_threads.append(helper_thread)  # Track the thread.

    def stop_helper_threads(self):
        print("[DecodingCommunicationHandler] stopping helper threads", flush=True)

        # Mark local threads as stopping first.
        self.stop_event.set()

        # Wake peer recv threads blocked in dist.recv / dist.broadcast.
        if hasattr(self, "tree_candidates_send_queues"):
            self.tree_candidates_send_queues.add(self._make_stop_payload("tree_candidates"))
        if hasattr(self, "tree_decoding_send_queues"):
            self.tree_decoding_send_queues.add(self._make_stop_payload("tree_decoding"))
        if self.if_last_rank and hasattr(self, "new_token_send_queues") and not self.round_end_sent:
            self.new_token_send_queues.add(self._make_round_end_payload())

        # Wake local send threads blocked on queue.remove.
        if hasattr(self, "tree_candidates_send_queues"):
            self.tree_candidates_send_queues.add(STOP_SIGNAL)
        if hasattr(self, "tree_decoding_send_queues"):
            self.tree_decoding_send_queues.add(STOP_SIGNAL)
        if hasattr(self, "new_token_send_queues"):
            self.new_token_send_queues.add(STOP_SIGNAL)

        alive_threads = []
        for thread in self.helper_threads:
            if thread.is_alive():
                thread.join(timeout=THREAD_JOIN_TIMEOUT_SEC)
            if thread.is_alive():
                alive_threads.append(str(thread))

        self.helper_threads.clear()

        if alive_threads:
            print(f"[DecodingCommunicationHandler] still alive after stop: {alive_threads}", flush=True)
        print("[DecodingCommunicationHandler] stopped", flush=True)

    def _make_stop_payload(self, tensor_name):
        return torch.zeros(
            self.tensor_shape_for_recv[tensor_name],
            dtype=self.tensor_type[tensor_name],
        )

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
    while True:
        tensor = send_queue.remove()
        if tensor is STOP_SIGNAL:
            break
        _broadcast_send(tensor, src)

def broadcast_recv_helper_thread(recv_queue, tensor_shape, src_rank, dtype, stop_event):
    print("[broadcast_recv_helper_thread] started", flush=True)

    while True:
        tensor = _broadcast_recv(tensor_shape, src_rank, dtype)
        point_id = int(tensor.view(-1)[0].item())
        if point_id == ROUND_END_POINT_ID:
            print("[broadcast_recv_helper_thread] round end received", flush=True)
            recv_queue.add(tensor)
            break
        recv_queue.add(tensor)
        if stop_event.is_set():
            break

def _broadcast_send(tensor, src_rank ):
    if tensor.device != torch.device("cpu"): # for gloo 
        tensor = tensor.cpu()
    dist.broadcast(tensor=tensor , src=src_rank)


def _broadcast_recv(tensor_shape, src_rank,dtype):
    tensor = torch.zeros(tensor_shape, dtype=dtype) 
    dist.broadcast(tensor, src=src_rank)
    return tensor
    
