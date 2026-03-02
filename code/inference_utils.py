import re
import torch
torch.cuda.empty_cache()


##just tags##
def annotate_tag(r):
    r = r.strip()
    if any(x in r for x in ['[[1]]', '[1]', '1']):
        if any(x in r for x in ['[[2]]', '[2]', '2']):
            return r # 1 and 2 (both)
        if any(x in r for x in ['[[1]]', '[1]', '1']):
            return 1 # only 1
    
    if any(x in r for x in ['[[2]]', '[2]', '2']):
        return 2
    else:
        return r

def parse_knowledge(text: str) -> dict:    
    def _normalize(s: str) -> str:
        s = re.sub(r'\r\n?', '\n', s)          
        s = re.sub(r'[ \t]+', ' ', s)     
        return s.strip()

    text = _normalize(text)


    ans_pattern = re.compile(
        r'Knowledge[_\s]*(\d+)\s*:\s*'          
        r'(.*?)\s*'                                             
        r'(?:\(|,)?\s*confidence[:\s]*([0-9]{1,3})\s*%?\s*\)?', # confidence
        re.IGNORECASE | re.DOTALL
    )

    
    final_answer_pattern = re.compile(
        r'^Final[_\s]+Answer\s*:\s*(.*)$',
        re.IGNORECASE | re.MULTILINE
    )
    

    final_confidence_pattern = re.compile(
        r'Overall[_\s]+Confidence.*?:\s*([0-9]{1,3})\s*%?',
        re.IGNORECASE
    )

    out = {
        "Answer_Knowledge": [],
        "Final_Answer": None,
        "Final_Confidence": None,
    }
   

    for num, know, conf in ans_pattern.findall(text):
        out["Answer_Knowledge"].append({
            f"Knowledge_{num}": know.strip(),
            "Confidence": int(conf),
        })

    final_ans = final_answer_pattern.search(text)
    if final_ans:
        out["Final_Answer"] = final_ans.group(1).strip()

    overall = final_confidence_pattern.search(text)
    if overall:
        out["Final_Confidence"] = int(overall.group(1))

    return out
