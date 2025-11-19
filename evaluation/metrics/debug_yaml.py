import torch
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 配置区域 =================
MODEL_PATH = "/newdata/LLMs/Llama-3-8B-Instruct"
DATA_PATH = "/data/home/yuanxiaosong/GAQ-Aware-Prune/data/zeroshot/piqa/validation.jsonl"
DEVICE = "cuda:2"  # 你的显卡
# ===========================================

def load_model():
    print(f"正在加载模型: {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # Llama-3 必须设置 pad_token，通常设为 eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map=DEVICE,
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer

def get_log_prob(model, tokenizer, context_text, choice_text):
    """
    计算给定 Context 下，Choice 的 Log Probability
    """
    # 1. 对 Context 进行编码（不加 Special Tokens，我们自己控制）
    # Llama-3 Instruct 的关键：使用 chat 模板格式化 context
    messages = [{"role": "user", "content": context_text}]
    # apply_chat_template 会自动添加 <|begin_of_text|>...<|eot_id|> 等
    # add_generation_prompt=True 会添加 <|start_header_id|>assistant<|end_header_id|>
    context_enc = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(model.device)
    
    # 2. 对 Choice 进行编码
    # 注意：Choice 不需要加 Special Tokens，它是纯文本补全
    choice_enc = tokenizer(choice_text, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    
    # 3. 拼接整个序列: Context + Choice
    input_ids = torch.cat([context_enc, choice_enc], dim=1)
    
    # 4. 前向传播计算 Loss
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        
    # 5. 提取 Choice 部分的 Log Prob
    # logits 的形状是 [1, seq_len, vocab_size]
    # 我们需要预测的是从 context_len 开始的 token
    # input_ids: [tok_c1, tok_c2, ..., tok_cn, tok_a1, tok_a2]
    # logits:    [pred_c2, ..., pred_cn, pred_a1, pred_a2, pred_next]
    
    # shift logits: 预测位置 i 的 logit 在 index i-1
    context_len = context_enc.shape[1]
    full_len = input_ids.shape[1]
    
    # 提取对应 Choice 那个片段的 logits
    # 我们关心的是 input_ids[context_len:] 这些 token 的概率
    # 它们对应的 logits 索引是 input_ids[context_len-1 : -1]
    
    target_ids = input_ids[:, context_len:]
    relevant_logits = logits[:, context_len-1 : -1, :]
    
    # 使用 CrossEntropyLoss(reduction='none') 来获取每个 token 的 loss (即 -log_prob)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    # view(-1) 展平处理
    token_losses = loss_fct(relevant_logits.reshape(-1, relevant_logits.size(-1)), target_ids.reshape(-1))
    
    # Sum token losses -> Total Negative Log Likelihood
    total_nll = token_losses.sum().item()
    
    # 返回 Log Likelihood (即负的 loss)
    return -total_nll

def evaluate():
    model, tokenizer = load_model()
    
    with open(DATA_PATH, 'r') as f:
        lines = f.readlines()
        
    correct = 0
    total = 0
    
    print(f"开始评估，共 {len(lines)} 条数据...")
    
    # 打印第一条数据的 Prompt 样子，用于调试
    first_sample = json.loads(lines[0])
    debug_msgs = [{"role": "user", "content": first_sample['goal']}]
    debug_prompt = tokenizer.apply_chat_template(debug_msgs, add_generation_prompt=True, tokenize=False)
    print(f"\n=== [调试] 实际输入给模型的 Prompt 格式 ===\n{debug_prompt}\n==========================================\n")

    for i, line in tqdm(enumerate(lines), total=len(lines)):
        item = json.loads(line)
        
        goal = item['goal']
        sol1 = item['sol1']
        sol2 = item['sol2']
        label = item['label'] # 0 or 1
        
        # 计算两个选项的分数
        score1 = get_log_prob(model, tokenizer, goal, sol1)
        score2 = get_log_prob(model, tokenizer, goal, sol2)
        
        # 预测
        prediction = 0 if score1 > score2 else 1
        
        if prediction == label:
            correct += 1
        total += 1
        
        # 每100条打印一次当前准确率
        if (i + 1) % 100 == 0:
            print(f" Step {i+1}: 当前准确率 = {correct/total:.2%}")

    print(f"\n最终结果: {correct}/{total} = {correct/total:.2%}")

if __name__ == "__main__":
    evaluate()