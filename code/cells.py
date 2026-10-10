import sys,re,collections
P=['8b7','4b2','3b9','2b8','2b2','1b5','1b1','619m','574m','421m','280m','221m','220m','196m','146m','83m','44m','25m','21m','14m','7m']
def v(t):
    m=re.fullmatch(r'(\d+)([mb])(\d?)',t); a=float(m.group(1)+('.'+m.group(3) if m.group(3) else '')); return a*(1e6 if m.group(2)=='m' else 1e9)
def parse(n):
    for p in P:
        if n.startswith(p):
            r=n[len(p):]
            for i in range(1,len(r)):
                a,b=r[:i],r[i:]
                if re.fullmatch(r'\d+[mb]\d?',a) and re.fullmatch(r'\d+[mb]\d?',b) and v(a)>=v(b)*0.99:
                    return v(p),v(a),v(b)
    return None
inv=set(l.split('\t')[0] for l in open(sys.argv[2]))
cells=collections.defaultdict(list); bad=[]
for line in open(sys.argv[1]):
    n,l=line.rstrip().split('\t')
    if float(l)<=0: continue  # 0 = run not finished (notebook comment)
    q=parse(n)
    if not q: bad.append(n); continue
    N,D,U=q; cells[(N,U)].append((D/U,float(l),n,n in inv))
print('unparsed',len(bad),bad[:20])
tot=0;rise=[]
for (N,U),xs in sorted(cells.items()):
    xs.sort(); 
    if len(xs)<2: continue
    eps=[round(e,1) for e,_,_,_ in xs]; ls=[l for _,l,_,_ in xs]
    i=min(range(len(ls)),key=lambda k:ls[k]); r=ls[-1]-ls[i]
    r64=max([l for e,l,_,_ in xs if e<=64 and e>=xs[i][0]]+[ls[i]])-ls[i]
    ck=sum(1 for x in xs if x[3])
    print(f'N={N/1e6:.0f}M U={U/1e6:.0f}M n={len(xs)} ckpt_in_misc={ck} epochs={eps} argmin_ep={xs[i][0]:.1f} minL={ls[i]:.3f} lastL={ls[-1]:.3f} rise_total={r:.3f} rise<=64ep={r64:.3f}')
