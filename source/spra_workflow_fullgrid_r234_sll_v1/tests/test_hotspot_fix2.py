from __future__ import annotations
import unittest
from joint18v2.common import token_hash
from wfsched.fixed_work import convert_nodes, INPUT_TOKENS, OUTPUT_TOKENS


def make_nodes(repeats=4, original_len=2900):
    prefix=[28]*32
    shared_hash=token_hash(prefix)
    nodes={}
    for w in range(repeats):
        for j in range(2):
            k=f'w{w:04d}/step{j+1}'
            tokens=prefix+[71+j]*(original_len-32)
            nodes[k]={'dataset_id':'public-same-packet', 'step_position':j,
                      'parents':([f'w{w:04d}/step1'] if j else []),
                      'share':32,'prefix_hash':shared_hash,'tokens':tokens,'wid':f'w{w:04d}'}
    return nodes


class HotspotRequestIdentity(unittest.TestCase):
    def test_repeated_hotspot_unique_native_hashes_and_exact_tokens(self):
        n=make_nodes()
        original={k:list(x['tokens']) for k,x in n.items()}
        receipt=convert_nodes(n,97)
        self.assertIsInstance(receipt,str)
        self.assertEqual(len({x['hash'] for x in n.values()}),len(n))
        self.assertEqual(sum(bool(x['fixed_native_hash_disambiguated']) for x in n.values()),6)
        for key,x in n.items():
            self.assertEqual(len(x['tokens']),INPUT_TOKENS)
            self.assertEqual(x['output'],OUTPUT_TOKENS)
            self.assertEqual(x['tokens'][:len(original[key])],original[key])
            self.assertEqual(token_hash(x['tokens'][:32]),x['prefix_hash'])
            self.assertEqual(token_hash(x['tokens']),x['hash'])

    def test_non_repeated_case_is_identical_to_old_padding(self):
        n=make_nodes(repeats=1)
        original={k:list(x['tokens']) for k,x in n.items()}
        convert_nodes(n,97)
        for k,x in n.items():
            self.assertEqual(x['tokens'],original[k]+[97]*(INPUT_TOKENS-len(original[k])))
            self.assertFalse(x['fixed_native_hash_disambiguated'])

    def test_rejects_collision_without_padding_space(self):
        n=make_nodes(repeats=2,original_len=INPUT_TOKENS)
        with self.assertRaisesRegex(ValueError,'no free .*padding'):
            convert_nodes(n,97)

    def test_repeated_group_fingerprint_independent_of_namespace(self):
        a,b=make_nodes(),make_nodes()
        for node in b.values():
            node['tokens'][3]=29
            node['prefix_hash']=token_hash(node['tokens'][:32])
        self.assertEqual(convert_nodes(a,97),convert_nodes(b,97))
