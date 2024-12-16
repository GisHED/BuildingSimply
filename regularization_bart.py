from transformers import BertTokenizer, BartForConditionalGeneration
from regularization_utitls import *
import matplotlib.pyplot as plt

def rearrange_list(nums):
    if type(nums) == str:
        nums = [int(i) for i in nums.split(' ')]
    if not nums:  # 如果列表为空，直接返回空列表
        return []

    nums = nums[:-1]
    # 找到最小值及其索引
    min_value = min(nums)
    min_index = nums.index(min_value)

    # 重新排序列表
    reordered = nums[min_index:] + nums[:min_index]

    reordered.append(reordered[0])
    reordered_str = ' '.join([str(i) for i in reordered])
    return reordered_str


class Regular_Bart():
    def __init__(self,start_id=None, maxlen=64):
        self.model = None
        self.load_model()


    def load_model(self,config_path=None,weight_path=None):
        token_path='./model_saved/regularization_bart/vocab.txt'
        self.tokenizer =  BertTokenizer.from_pretrained(token_path, do_lower_case=True,max_len=64)
        model_path = './model_saved/regularization_bart/' if not weight_path else weight_path
        self.model = BartForConditionalGeneration.from_pretrained(model_path,ignore_mismatched_sizes=True)


    def generate(self, vec, topk=3, topp=0.95):
        vec = rearrange_list(vec)
        input_ids = self.tokenizer.encode(vec, return_tensors='pt')
        pred_ids = self.model.generate(input_ids, num_beams=2, max_length=64)
        # 将 tensor 转换为列表
        predicted_list = pred_ids.tolist()[0]
        
        # 找到第一个出现的结束 token 的位置
        try:
            end_idx = predicted_list.index(-1)
        except ValueError:
            end_idx = len(predicted_list)
        
        # 提取序列，跳过开始 token（第一个 3）和之后的结束 token
        extracted_sequence = [token_id for token_id in predicted_list[1:end_idx] if token_id != 101 and token_id != 102]
        
        # 打印结果
        # print(extracted_sequence)
        # print(tokenizer.convert_ids_to_tokens(extracted_sequence))  
        result_list = self.tokenizer.convert_ids_to_tokens(extracted_sequence)
        return ' '.join(result_list)

def argument_vec_pts(vec_p,flag=True):
    argument_vec = argument_by_all_index(vec_p,list(range(len(vec_p))[1:])) if flag else vec_p
    argument_vec = argument_by_PIL(argument_vec)
    return argument_vec

class load_data:
    def __init__(self,sample_num=8) -> None:
        datafile_name = "database/regular_urban.json"
        try:
            self.vecs = read_json(datafile_name)
        except:
            self.vecs = read_txt(datafile_name)
        self.load_source_vecpts(sample_num)



    def load_source_vecpts(self,sample_num=8):
        vecs = self.vecs
        vecs = [vec for vec in vecs if len(vec) > 8 ]
        vec_array = np.random.choice(vecs,sample_num)
        vec_p = [vec2point(vec) for vec in vec_array]
        argument_vec = [argument_vec_pts(v,1) for v in vec_p]
        self.source_vecpts = argument_vec

if __name__ == '__main__':
    vec_data = load_data()

    regular_urban = Regular_Bart()
    show_vec_data = []
    for vec in vec_data.source_vecpts:
        vec = serialize(vec)
        vec = vec['indexs']        
        test_vec = ' '.join([str(i) for i in vec])
        show_vec_data.append([int(i) for i in test_vec.split(' ')])
        result_vec = regular_urban.generate(test_vec)
        sv = [int(i) for i in result_vec.split(' ')]
        show_vec_data.append(sv)
    show_vec_list(show_vec_data)
    a = input()
