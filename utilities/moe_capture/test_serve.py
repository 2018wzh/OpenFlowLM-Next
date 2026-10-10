"""Incremental tests for compact native-value SSE projections."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer, models
from serve import Engine


class PacketTests(unittest.TestCase):
    def setUp(self):
        self.engine = Engine.__new__(Engine)
        self.engine.tokenizer = Tokenizer(models.WordLevel({'[UNK]':0,'bank':1,'next':2},unk_token='[UNK]'))

    def packet(self, kind, values, **fields):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            descriptors=[]; raw=bytearray()
            for name,value,slot in values:
                value=np.asarray(value,dtype='<i4' if name=='expert_ids' else '<f4')
                descriptor=dict(name=name,dtype='i32' if name=='expert_ids' else 'f32',shape=list(value.shape),
                                offset=len(raw),length=value.nbytes,source='native_openflowlm',event_seq=12)
                if slot is not None: descriptor['slot']=slot
                descriptors.append(descriptor); raw.extend(value.tobytes())
            (directory/'tensors.bin').write_bytes(raw)
            event=dict(kind=kind,event_seq=12,input_token_id=1,phase='decode',tensors=descriptors,**fields)
            return self.engine.packet(event,directory,'run',False)

    def test_topk_uses_dispatched_order_and_native_values(self):
        ids=[9,4,7,2,8,1,5,3]
        logits=np.arange(256,dtype=np.float32)*.01
        probabilities=np.full(256,.00390625,dtype=np.float32)
        weights=np.array([.3,.2,.15,.1,.09,.07,.05,.04],dtype=np.float32)
        p=self.packet('route',[('expert_ids',ids,None),('router_logits',logits,None),
                             ('router_probabilities',probabilities,None),('expert_weights',weights,None)])
        self.assertEqual([v['id'] for v in p['routing']['topk']],ids)
        self.assertEqual(p['routing']['topk'][0]['logit'],float(logits[9]))
        self.assertEqual(p['routing']['topk'][0]['prob'],float(probabilities[9]))
        self.assertEqual(p['routing']['topk'][0]['weight'],float(weights[0]))
        self.assertEqual(p['display'],'bank')
        self.assertTrue(all('data' not in t for t in p['tensors']))

    def test_logits_distinguish_input_and_next_token(self):
        p=self.packet('logits',[('logits',[0,1,2],None)],next_token_id=2)
        self.assertEqual(p['display'],'bank')
        self.assertEqual(p['next_token_display'],'next')
        self.assertEqual(p['generated_text'],'next')

    def test_shared_branch_norm_includes_gate(self):
        values=[('expert_output',[3,4],i) for i in range(8)]
        values += [('shared_output',[3,4],8),('shared_gate',[.25],None),
                   ('routed_mixture',[6,8],None),('residual_after',[0,2],None)]
        p=self.packet('moe_output',values)
        self.assertEqual(p['norms']['shared_output_l2'],5)
        self.assertEqual(p['norms']['gated_shared_output_l2'],1.25)
        self.assertEqual(p['norms']['routed_mixture_l2'],10)
        self.assertEqual(p['norms']['source'],'derived')


if __name__=='__main__': unittest.main()
