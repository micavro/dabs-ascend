"""Copied synthetic fixture/tokenizer from supplied typed19 tests; NOT real inference."""
def fixture():
    return {'id':'SYNTHETIC_TEST','question':'DO_NOT_SEND_OVERALL', 'answerable':True,
      'answer':'GOLD_SECRET','answer_aliases':[],
      'paragraphs':[{'idx':0,'title':'City','paragraph_text':'Paris is a city. Sam lives in Paris.','is_supporting':True},
        {'idx':1,'title':'Other','paragraph_text':'Other material. '*15,'is_supporting':False}],
      'question_decomposition':[{'id':77,'question':'Which city?','answer':'Paris','paragraph_support_idx':0},
        {'id':88,'question':'Which person?','answer':'Sam','paragraph_support_idx':0},
        {'id':99,'question':'Where does #2 live in #1?','answer':'Paris','paragraph_support_idx':0}]}

class FakeTokenizer:
    markers={'<|im_start|>':151644,'<|im_end|>':151645,'system':900001,'user':900002,'assistant':900003}
    def __init__(self):self.calls=self.decode_calls=self.chat_calls=0;self.first_blocks=set();self.sent=[]
    async def encode(self,text):
        self.calls+=1;out=[];i=0
        while i<len(text):
            match=next((x for x in self.markers if text.startswith(x,i)),None)
            if match:out.append(self.markers[match]);i+=len(match)
            else:out.append(ord(text[i])+100);i+=1
        return out or [132]
    async def decode(self,tokens):
        self.decode_calls+=1;r={v:k for k,v in self.markers.items()}
        return ''.join(r[t] if t in r else chr(t-100) for t in tokens)
    async def chat_tokens(self,system,user):
        self.chat_calls+=1;self.sent.append((system,user))
        return await self.encode('<|im_start|>system\n'+system+'<|im_end|>\n<|im_start|>user\n'+user+'<|im_end|>\n<|im_start|>assistant\n')

