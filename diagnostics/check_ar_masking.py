"""Does replacing mutated positions with 'X' change anything, given the autoregressive mask?

Two mechanisms are at work and they are independent:
  (a) the AR decoding mask -- position i attends only to positions decoded BEFORE it, so
      logits[i] cannot depend on S[i];
  (b) the explicit 'X' substitution at mutated positions in the INPUT sequence.
If (a) alone were doing the work, (b) would be a no-op. Test it directly: same structure, same
decoding order, S = mutant vs S = mutant-with-X, and compare logits at the mutated positions
against logits everywhere else.
"""
import sys, torch, numpy as np
sys.path.insert(0, '/home/guoj0f/repos/BindingGYM/baselines/protein_mpnn')
from protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize

R='/home/guoj0f/repos/H3-DDG/.claude/worktrees/reproduce'
dev=torch.device('cuda')
m=ProteinMPNN(num_letters=21,node_features=128,edge_features=128,hidden_dim=128,
              num_encoder_layers=3,num_decoder_layers=3,augment_eps=0.0,k_neighbors=48,ca_only=False)
m.load_state_dict(torch.load('/home/guoj0f/repos/BindingGYM/training/cache/v_48_020.pt',
                             map_location='cpu')['model_state_dict']); m.to(dev).eval()

d=parse_PDB(f'{R}/data/input/structures/1BE9_hm.pdb', ca_only=False)
ch=sorted([k[-1:] for k in d[0] if k[:9]=='seq_chain'])
b=tied_featurize(d,'cpu',{d[0]['name']:(ch,[])},None,None,None,None,None,ca_only=False)
X,S,mask,chain_M,residue_idx,chain_enc = b[0],b[1],b[2],b[4],b[12],b[5]
X,S,mask,chain_M,residue_idx,chain_enc=[t.to(dev) for t in (X,S,mask,chain_M,residue_idx,chain_enc)]

# 造 3 个突变位点
torch.manual_seed(0)
pos=[40]   # 单点：排除「突变位点互相看见」的干扰
S_mut=S.clone()
for p in pos: S_mut[0,p]=(S_mut[0,p]+3)%20
S_masked=S_mut.clone()
for p in pos: S_masked[0,p]=20            # 'X'

randn=torch.randn(chain_M.shape,device=dev)   # 同一个解码顺序，两次调用共用
with torch.no_grad():
    lp_mut    = m(X,S_mut,   mask,chain_M,residue_idx,chain_enc,randn)
    lp_masked = m(X,S_masked,mask,chain_M,residue_idx,chain_enc,randn)

order = torch.argsort((chain_M+0.0001)*torch.abs(randn))[0]
rank  = torch.empty_like(order); rank[order]=torch.arange(len(order),device=dev)
diff  = (lp_mut-lp_masked).abs().max(dim=-1).values[0]

print(f"结构 1BE9，长度 {S.shape[1]}，人工突变位点 {pos}（解码顺序中的名次 {[int(rank[p]) for p in pos]}）\n")
print(f"{'位点类别':28} {'max|Δlogprob|':>16}")
print(f"{'突变位点自身':28} {diff[pos].max().item():>16.3e}")
after  = [i for i in range(len(diff)) if i not in pos and rank[i] > min(rank[p] for p in pos)]
before = [i for i in range(len(diff)) if i not in pos and rank[i] < min(rank[p] for p in pos)]
print(f"{'解码顺序在最早突变位点之后':28} {diff[after].max().item():>16.3e}   ({len(after)} 个位点)")
print(f"{'解码顺序在最早突变位点之前':28} {diff[before].max().item():>16.3e}   ({len(before)} 个位点)")
print(f"\n受影响位点数（|Δ|>1e-5）: {(diff>1e-5).sum().item()} / {len(diff)}")
