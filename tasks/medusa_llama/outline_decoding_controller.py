from jupiter.core.threadsafe_queue import Queue
from .kv_cache import initialize_past_key_values
import os
class OutlineDecodingController:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(OutlineDecodingController, cls).__new__(cls)
        return cls._instance
    def __init__(self, points, config, model):
        if not hasattr(self, 'initialized'):
            self.points = points
            self.point_num = len(points)
            self.config = config
            self.model = model
            self.initialized = True 
            point_kv_env = os.getenv("JUPITER_POINT_KV_CACHE_LENGTH")
            if point_kv_env is not None:
                self.point_kv_cache_length = int(point_kv_env)
            elif hasattr(self.config, "max_point_kv_cache_length"):
                self.point_kv_cache_length = int(self.config.max_point_kv_cache_length)
            elif hasattr(self.config, "max_kv_cache_length"):
                self.point_kv_cache_length = min(int(self.config.max_kv_cache_length), 256)
            else:
                self.point_kv_cache_length = min(int(self.config.max_position_embeddings), 256)
            # points kv cache
            self.past_key_values_for_point = []
            self.past_key_values_data_for_point = []
            self.current_length_data_for_point = []
            # input_ids for points
            self.input_ids_for_point = []
            # input_len for points
            self.input_len_for_point = []
            self.shared_prefix_len_for_point = []
            # set up request queue
            if self.config.is_last_stage:
                self.request_queue = Queue()
            # record whether point is finish
            self.is_finish = [False]*self.point_num
            self.prepare_point_kv_cache()
            self.debug_kv = os.getenv("JUPITER_KV_DEBUG", "1") == "1"
    def set_up_input_ids_for_point(self, input_ids_for_point):
        self.input_ids_for_point = input_ids_for_point
        self.input_len_for_point = [ input_ids.shape[1] for input_ids in input_ids_for_point]
        self.shared_prefix_len_for_point = []
        for point_id, input_ids in enumerate(input_ids_for_point):
            point_current_len = int(self.current_length_data_for_point[point_id].max().item())
            shared_prefix_len = int(input_ids.shape[1]) - point_current_len
            if shared_prefix_len < 0:
                raise RuntimeError(
                    f"[KV][init] invalid shared prefix length for point={point_id}: "
                    f"input_ids_len={int(input_ids.shape[1])}, point_current_len={point_current_len}, "
                    f"shared_prefix_len={shared_prefix_len}"
                )
            self.shared_prefix_len_for_point.append(shared_prefix_len)
            if self.debug_kv:
                print(
                    f"[KV][init] point={point_id} input_ids_len={int(input_ids.shape[1])} "
                    f"point_current_len={point_current_len} shared_prefix_len={shared_prefix_len}",
                    flush=True,
                )
    def add_request(self, medusa_logits, logits, point_id):
        assert self.config.is_last_stage
        if self.is_finish[point_id]:
            if self.debug_kv:
                print(f"[KV][queue] skip finished point={point_id}", flush=True)
            return
        if self.debug_kv:
            cur_len = int(self.current_length_data_for_point[point_id].max().item())
            print(f"[KV][queue] add point={point_id} current_len={cur_len}", flush=True)
        self.request_queue.add({
                        "point_id": point_id,
                        "medusa_logits": medusa_logits,
                        "logits": logits        
        })
    def add_requests(self, medusa_logits_list, logits_list):
        assert self.config.is_last_stage
        print("=====================\n init request: ", self.point_num)
        for point_id in range (self.point_num):
            self.add_request(medusa_logits_list[point_id],logits_list[point_id], point_id )
    def get_request(self ):
        assert self.config.is_last_stage
        request = self.request_queue.remove()  
        return request 
    def get_active_request(self, max_kv_cache_len, tree_decode_window):
        assert self.config.is_last_stage
        if max_kv_cache_len is None:
            effective_limit = self.point_kv_cache_length
        else:
            effective_limit = min(int(max_kv_cache_len), int(self.point_kv_cache_length))
        while True:
            if self.all_points_finish():
                return None
            request = self.request_queue.remove()
            point_id = request["point_id"]
            if self.is_finish[point_id]:
                if self.debug_kv:
                    print(f"[KV][dispatch] skip finished point={point_id}", flush=True)
                continue
            if effective_limit is None:
                if self.debug_kv:
                    cur_len = int(self.current_length_data_for_point[point_id].max().item())
                    print(f"[KV][dispatch] select point={point_id} current_len={cur_len} no_limit", flush=True)
                return request
            current_len = int(self.current_length_data_for_point[point_id].max().item())
            if current_len >= (effective_limit - tree_decode_window):
                self.is_finish[point_id] = True
                if self.debug_kv:
                    print(
                        f"[KV][dispatch] mark_finish point={point_id} current_len={current_len} "
                        f"limit={effective_limit} window={tree_decode_window}",
                        flush=True,
                    )
                continue
            if self.debug_kv:
                print(
                    f"[KV][dispatch] select point={point_id} current_len={current_len} "
                    f"limit={effective_limit} window={tree_decode_window}",
                    flush=True,
                )
            return request
    def prepare_point_kv_cache(self):
        print("=====================\n prepare point kv cache")
        print(f"[KV][init] point_kv_cache_length={self.point_kv_cache_length}", flush=True)
        for _ in range(self.point_num):
            past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(
                self.model,
                max_length_override=self.point_kv_cache_length,
            )
            self.past_key_values_for_point.append(past_key_values)
            self.past_key_values_data_for_point.append(past_key_values_data)
            self.current_length_data_for_point.append(current_length_data)

    def get_point_past_key_values_data(self,point_id):
        return self.past_key_values_data_for_point[point_id]
    def get_point_past_key_values(self,point_id ):
        return self.past_key_values_for_point[point_id]
    def get_point_current_length_data(self,point_id):
        return self.current_length_data_for_point[point_id]
    def get_input_ids(self,point_id):
        return self.input_ids_for_point[point_id]
    def update_input_ids(self,  input_ids, point_id):
        self.input_ids_for_point[point_id] = input_ids
    def get_input_len(self,point_id):
        return self.input_len_for_point[point_id]
    def get_shared_prefix_len(self, point_id):
        return self.shared_prefix_len_for_point[point_id]
    def get_point_kv_cache_length(self):
        return self.point_kv_cache_length
    def set_point_finish(self, point_id):
        self.is_finish[point_id] = True
    def is_point_finish(self, point_id):
        return self.is_finish[point_id]
    def all_points_finish(self):
        return all(self.is_finish)

    def get_output(self ):
        tokenizer = self.model.get_tokenizer()
        for i in range(self.point_num):
            input_len = self.input_len_for_point[i]
            input_ids = self.input_ids_for_point[i]
            text = tokenizer.decode(
                        input_ids[0, input_len :],
                        skip_special_tokens=True,
                        spaces_between_special_tokens=False,
                        clean_up_tokenization_spaces=True,
                    ) 
            print("********************\n",flush=True)
            print(self.points[i] + text,flush=True)
    def release(self):
        # Release references so CUDA memory can be reclaimed.
        self.past_key_values_for_point = []
        self.past_key_values_data_for_point = []
        self.current_length_data_for_point = []
        self.input_ids_for_point = []
        self.input_len_for_point = []
        self.shared_prefix_len_for_point = []
        self.is_finish = []
        if hasattr(self, "request_queue"):
            self.request_queue = None
        self.points = []
        self.point_num = 0


controller = None
def get_controller():
    return controller
def set_controller(con):
    global controller
    controller = con


def reset_controller():
    global controller
    if controller is not None and hasattr(controller, "release"):
        controller.release()
    controller = None
    OutlineDecodingController._instance = None
