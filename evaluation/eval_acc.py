from typing import List
import re
import string


def get_exact_match(generations:list, answers:list) -> List[float]:
    '''
    generations: list of single generation -> List[str]
    answers: list of list of answers -> List[List[str]]

    return: list of exact match score -> List[float]
    '''
    exact_match = []
    for i in range(len(generations)):
        exact_match_i = []
        
        gen = _normalize_text(generations[i]) 
        
        for j in range(len(answers[i])):
            ans = _normalize_text(answers[i][j]) 
        
            if ans == gen:
                exact_match_i.append(1.0)
            else:
                exact_match_i.append(0.0)
            
            if exact_match_i[-1] == 1.0:
                break
        
        if len(exact_match_i) == 0:
            exact_match.append(0.0)
        else:
            exact_match.append(max(exact_match_i))
    
    return exact_match


def _normalize_text(s:str, lower_bool:bool=True)->str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    
    if lower_bool:
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    else:
        return white_space_fix(remove_articles(remove_punc(s)))