"""Full third-cumulant history with Gaussian background pruning.

The factor columns are packed to reduce Python overhead. The radial correction
is a formal extrapolation calibrated on public development data, not a claim
that the intermediate state is a realizable spherical distribution.
"""
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator

def _unpack_large(batch,m,k,n):
    return batch*(m*k+k*n)>=27000000

def _assemble_quadrants(p1,p2,p3,p4,p5,p6,p7,out=None):
    batch,mh,nh=p1.shape
    if out is None and 4*batch*mh*nh<2097152:
        # Small outputs: the direct block keeps fewer array operations; large outputs are written in place.
        u1=p1+p2
        u2=p1+p6
        u3=u2+p7
        u4=u2+p5
        return fnp.block([[u1,u4+p3],[u3-p4,u3+p5]])
    result=fnp.empty((batch,2*mh,2*nh),dtype=p1.dtype) if out is None else out
    top,bottom=fnp.split(result,2,axis=1)
    r11,r12=fnp.split(top,2,axis=2)
    r21,r22=fnp.split(bottom,2,axis=2)
    fnp.add(p1,p2,out=r11)
    u2=p1+p6
    u3=u2+p7
    u4=u2+p5
    fnp.add(u4,p3,out=r12)
    fnp.subtract(u3,p4,out=r21)
    fnp.add(u3,p5,out=r22)
    return result

def _batched(a,b,depth,out=None):
    if depth==0:
        if out is not None:
            fnp.matmul(a,b,out=out)
            return out
        return a @ b
    batch,m,k=a.shape
    n=b.shape[-1]
    if not batch or not m or not k or not n:
        return a @ b
    mh,kh,nh=m//2,k//2,n//2
    # Bound each temporary to 104 MB under the server's 100 MiB array limit.
    per_batch=max(m*k,k*n,m*n) if depth==1 else 7*max(mh*kh,kh*nh,mh*nh)
    chunk=max(1,26000000//per_batch)
    if batch>chunk:
        if batch*m*n<=26000000 and 4*chunk*(m//2)*(n//2)>=2097152:
            # Chunked products write each chunk straight into one output buffer (no final concatenation).
            result=fnp.empty((batch,m,n),dtype=a.dtype) if out is None else out
            for i in range(0,batch,chunk):
                _batched(a[i:i+chunk],b[i:i+chunk],depth,out=result[i:i+chunk])
            return result
        return fnp.concatenate([_batched(a[i:i+chunk],b[i:i+chunk],depth) for i in range(0,batch,chunk)],axis=0)
    at,ab=fnp.split(a,2,axis=1)
    a11,a12=fnp.split(at,2,axis=2)
    a21,a22=fnp.split(ab,2,axis=2)
    bt,bb=fnp.split(b,2,axis=1)
    b11,b12=fnp.split(bt,2,axis=2)
    b21,b22=fnp.split(bb,2,axis=2)
    if depth>=2 and not (depth==2 and _unpack_large(batch,m,k,n)) and 4*batch*(mh*kh+kh*nh)>=4194304:
        # Large packed levels: write the seven operand blocks straight into the packed buffers
        # (order a22,a11,a12,s4,s1,s2,s3 / t4,b11,b21,b22,t1,t2,t3), same sums and products as below.
        left=fnp.empty((7*batch,mh,kh),dtype=a.dtype)
        lu,l3,l4,l5,l6=fnp.split(left,[3*batch,4*batch,5*batch,6*batch],axis=0)
        fnp.concatenate((a22,a11,a12),axis=0,out=lu)
        fnp.add(a21,a22,out=l4)
        fnp.subtract(l4,a11,out=l5)
        fnp.subtract(a11,a21,out=l6)
        fnp.subtract(a12,l5,out=l3)
        right=fnp.empty((7*batch,kh,nh),dtype=b.dtype)
        r0,ru,r4,r5,r6=fnp.split(right,[batch,4*batch,5*batch,6*batch],axis=0)
        fnp.concatenate((b11,b21,b22),axis=0,out=ru)
        fnp.subtract(b12,b11,out=r4)
        fnp.subtract(b22,r4,out=r5)
        fnp.subtract(b22,b12,out=r6)
        fnp.subtract(r5,b21,out=r0)
        products=_batched(left,right,depth-1)
        p4,p1,p2,p3,p5,p6,p7=fnp.split(products,7,axis=0)
        return _assemble_quadrants(p1,p2,p3,p4,p5,p6,p7,out=out)
    s1=a21+a22
    s2=s1-a11
    s3=a11-a21
    s4=a12-s2
    t1=b12-b11
    t2=b22-t1
    t3=b22-b12
    t4=t2-b21
    if depth==1:
        p1=a11 @ b11
        p2=a12 @ b21
        p3=s4 @ b22
        p4=a22 @ t4
        p5=s1 @ t1
        p6=s2 @ t2
        p7=s3 @ t3
        return _assemble_quadrants(p1,p2,p3,p4,p5,p6,p7,out=out)
    if depth==2 and _unpack_large(batch,m,k,n):
        p1=_batched(a11,b11,depth-1)
        p2=_batched(a12,b21,depth-1)
        p3=_batched(s4,b22,depth-1)
        p4=_batched(a22,t4,depth-1)
        p5=_batched(s1,t1,depth-1)
        p6=_batched(s2,t2,depth-1)
        p7=_batched(s3,t3,depth-1)
        return _assemble_quadrants(p1,p2,p3,p4,p5,p6,p7,out=out)
    left=fnp.concatenate((a11,a12,s4,a22,s1,s2,s3),axis=0)
    right=fnp.concatenate((b11,b21,b22,t4,t1,t2,t3),axis=0)
    products=_batched(left,right,depth-1)
    p1,p2,p3,p4,p5,p6,p7=fnp.split(products,7,axis=0)
    return _assemble_quadrants(p1,p2,p3,p4,p5,p6,p7,out=out)


def _padded_matmul(a,b,depth=2,upgrade=True):
    m,k=a.shape
    n=b.shape[1]
    if not m or not k or not n:
        return a @ b
    if depth==4 and upgrade:depth=5
    block=2**depth
    mp,kp,npad=(-m)%block,(-k)%block,(-n)%block
    mh,kh,nh=(m+mp)//2,(k+kp)//2,(n+npad)//2
    if 7*max(mh*kh,kh*nh,mh*nh)>26000000:
        return a @ b
    if mp or kp:a=fnp.pad(a,((0,mp),(0,kp)))
    if kp or npad:b=fnp.pad(b,((0,kp),(0,npad)))
    return _batched(a[None,:,:],b[None,:,:],depth)[0,:m,:n]

"""Integer operation-count bookkeeping for dimension-only product selection."""


def _shape_batch_cost(batch, m, k, n, depth, cache):
    key = (batch, m, k, n, depth)
    if key in cache:
        return cache[key]
    if not batch or not m or not k or not n:
        return 0
    if depth == 0:
        return batch * m * n * (2*k-1)
    mh, kh, nh = m//2, k//2, n//2
    per = max(m*k, k*n, m*n) if depth == 1 else 7*max(mh*kh, kh*nh, mh*nh)
    chunk = max(1, 26000000//per)
    if batch > chunk:
        whole, tail = divmod(batch, chunk)
        value = whole*_shape_batch_cost(chunk, m, k, n, depth, cache)
        if tail:
            value += _shape_batch_cost(tail, m, k, n, depth, cache)
        value += batch*m*n
    else:
        unpack = depth == 2 and batch*(m*k+k*n) >= 27000000
        inputs = 4 if depth == 1 or unpack else 11
        value = batch*(inputs*(mh*kh+kh*nh)+11*mh*nh)
        value += (7*_shape_batch_cost(batch, mh, kh, nh, depth-1, cache) if unpack
                  else _shape_batch_cost(7*batch, mh, kh, nh, depth-1, cache))
    cache[key] = value
    return value


def _shape_padded_cost(m, k, n, depth, cache):
    block = 2**depth
    M, K, N = [v+(-v)%block for v in (m, k, n)]
    if 7*max(M*K//4, K*N//4, M*N//4) > 26000000:
        return m*n*(2*k-1)
    value = _shape_batch_cost(1, M, K, N, depth, cache)
    if M != m or K != k:
        value += M*K
    if K != k or N != n:
        value += K*N
    return value


def _shape_product_cost(m, k, n, depth):
    if not m or not k or not n:
        return 0
    block = 2**depth
    M, K, N = [v+(-v)%block for v in (m, k, n)]
    if 7*max(M*K//4, K*N//4, M*N//4) > 26000000:
        return m*n*(2*k-1)
    mm, kk, nn = [v-v%block if v >= 2*block and 0 < v%block <= block//2 else v
                  for v in (m, k, n)]
    value = _shape_padded_cost(mm, kk, nn, depth, {})
    if kk < k:
        value += mm*nn*2*(k-kk)
    if nn < n:
        value += mm*(n-nn)*(2*k-1)+mm*n
    if mm < m:
        value += (m-mm)*n*(2*k-1)+m*n
    return value


def _profitable_depth6(m, k, n):
    return _shape_product_cost(m, k, n, 6)*100 <= _shape_product_cost(m, k, n, 5)*97


def matmul(a,b,depth=2,upgrade=True):
    m,k=a.shape
    n=b.shape[1]
    if not m or not k or not n:return a @ b
    if depth==4 and upgrade:depth=5
    if depth==5 and min(m,k,n)>=256 and (_profitable_depth6(m,k,n) or (_shape_product_cost(m,k,n,5)-_shape_product_cost(m,k,n,6))>=250000.0*max(1,_shape_product_calls(m,k,n,6)-_shape_product_calls(m,k,n,5))):depth=6
    block=2**depth
    M=m+(-m)%block
    K=k+(-k)%block
    N=n+(-n)%block
    if 7*max((M//2)*(K//2),(K//2)*(N//2),(M//2)*(N//2))>26000000:
        return a @ b
    # Peel only narrow fringes; the remaining dimensions use the original
    # bounded, padded product. The decision uses dimensions alone.
    mm=m-m%block if m>=2*block and 0<m%block<=block//2 else m
    kk=k-k%block if k>=2*block and 0<k%block<=block//2 else k
    nn=n-n%block if n>=2*block and 0<n%block<=block//2 else n
    if (mm,kk,nn)==(m,k,n):return _padded_matmul(a,b,depth=depth,upgrade=False)
    core=_padded_matmul(a[:mm,:kk],b[:kk,:nn],depth=depth,upgrade=False)
    if kk<k:core=core+a[:mm,kk:] @ b[kk:,:nn]
    if nn<n:core=fnp.concatenate((core,a[:mm,:] @ b[:,nn:]),axis=1)
    if mm<m:core=fnp.concatenate((core,a[mm:,:] @ b),axis=0)
    return core

def _shape_batch_calls(batch,m,k,n,depth,cache):
    key=(batch,m,k,n,depth)
    if key in cache:return cache[key]
    if depth==0 or not min(batch,m,k,n):return 1
    mh,kh,nh=m//2,k//2,n//2
    per=max(m*k,k*n,m*n) if depth==1 else 7*max(mh*kh,kh*nh,mh*nh)
    chunk=max(1,26000000//per)
    if batch>chunk:
        whole,tail=divmod(batch,chunk)
        value=whole*_shape_batch_calls(chunk,m,k,n,depth,cache)+1
        if tail:value+=_shape_batch_calls(tail,m,k,n,depth,cache)
    elif depth==1:value=29
    elif depth==2 and _unpack_large(batch,m,k,n):value=22+7*_shape_batch_calls(batch,mh,kh,nh,depth-1,cache)
    else:value=25+_shape_batch_calls(7*batch,mh,kh,nh,depth-1,cache)
    cache[key]=value
    return value

def _shape_padded_calls(m,k,n,depth):
    if not min(m,k,n):return 1
    block=2**depth
    M,K,N=[v+(-v)%block for v in (m,k,n)]
    if 7*max(M*K//4,K*N//4,M*N//4)>26000000:return 1
    return int(M!=m or K!=k)+int(K!=k or N!=n)+_shape_batch_calls(1,M,K,N,depth,{})

def _shape_product_calls(m,k,n,depth):
    if not min(m,k,n):return 1
    block=2**depth
    M,K,N=[v+(-v)%block for v in (m,k,n)]
    if 7*max(M*K//4,K*N//4,M*N//4)>26000000:return 1
    mm,kk,nn=[v-v%block if v>=2*block and 0<v%block<=block//2 else v for v in (m,k,n)]
    return _shape_padded_calls(mm,kk,nn,depth)+2*sum([mm<m,kk<k,nn<n])

def joint_projection_profitable(band_rows,active_rows,k,columns):
    total=band_rows+active_rows
    if not min(band_rows,active_rows,k,columns) or k*columns>26000000 or total*columns>26000000:return False
    da=6 if min(active_rows,k,columns)>=256 and _profitable_depth6(active_rows,k,columns) else 5
    dj=6 if min(total,k,columns)>=256 and _profitable_depth6(total,k,columns) else 5
    separate=_shape_product_cost(active_rows,k,columns,da)+band_rows*columns*(2*k-1)
    joint=_shape_product_cost(total,k,columns,dj)+total*k
    return joint<separate and _shape_product_calls(total,k,columns,dj)<=_shape_product_calls(active_rows,k,columns,da)+1

def compact_product(a,b):
    if min(a.shape[0],a.shape[1],b.shape[1])>=128:
        return matmul(a,b,depth=4,upgrade=False)
    return a @ b

def paired_projection(q,x,y):
    both=compact_product(q,fnp.concatenate((x,y),axis=1))
    return both[:,:x.shape[1]],both[:,x.shape[1]:]

def paired_readout(left,right,x,y):
    return compact_product(fnp.concatenate((left,right),axis=1),fnp.concatenate((y.T,x.T),axis=0))

def fourth_traces(x,y,f):
    """(shadow, tree) fourth-order traces as one 2-vector.  f=(2,4,k) weights on the per-column terms
    (xy*yy, xx*xx, xx*yy, xy*xy): shadow = a4*xy*yy + c4*xx^2, tree = d4*xx^2 + (b4/3)*(xx*yy + 2*xy^2)."""
    xx=fnp.einsum('ij,ij->j',x,x)
    yy=fnp.einsum('ij,ij->j',y,y)
    xy=fnp.einsum('ij,ij->j',x,y)
    return fnp.einsum('mkj,kj->m',f,fnp.stack((xy*yy,xx*xx,xx*yy,xy*xy)))

def identity_fourth_traces(y,xy,f):
    """fourth_traces for x = coordinate columns: xx = 1 and xy = the selected diagonal of y (both exact)."""
    yy=fnp.einsum('ij,ij->j',y,y)
    return fnp.einsum('mkj,kj->m',f,fnp.stack((xy*yy,fnp.ones_like(yy),yy,xy*xy)))

def trace_weights(cm):
    return cm[6:14].reshape((2,4,cm.shape[1]))

def _fill(x,v):
    # x + diag(v - diag(x)) for a freshly computed square x, written in place.
    fnp.fill_diagonal(x,v)
    return x

def coefficient_views(ac,bc,cc,a4,c4,b4,d4,scale=None):
    if scale is not None:
        # Same rows as below, built as one stack times a per-row constant (zero rows from ac*0, ac>=0).
        return fnp.stack((ac,bc,bc,cc,ac,fnp.ones_like(ac),a4,c4,ac,ac,ac,d4,b4,b4,b4,cc))*scale
    # One stacked coefficient matrix per block (columns = block columns): rows 0:2 left mix (2ac,bc),
    # 2:6 right mix (2bc,3cc,ac,2), 6:14 fourth-trace weights [[a4,c4,0,0],[0,d4,b4/3,2b4/3]], 14 b4, 15 cc.
    zero=fnp.zeros_like(ac)
    b3=b4/3
    return fnp.stack((2*ac,bc,2*bc,3*cc,ac,fnp.full_like(ac,2),a4,c4,zero,zero,zero,d4,b3,2*b3,b4,cc))

def _restore_band(band,btd,pc_g2,g3_full,g4_full,bmetric,radial3,pm,pm_raw2,pc,fulln):
    """Gaussian-closure mean/variance for band units (unchanged formulas)."""
    bg2=pc_g2[band]
    bg3=g3_full[band]
    bg4=g4_full[band]
    bkd=radial3*bmetric*bmetric
    bgmu=pm[band]
    restored=bgmu+btd*bg3/6+bkd*bg4/24
    br2=pm_raw2[band]+btd*bg2/3+bkd*bg3/12
    restored_var=fnp.maximum(br2-restored*restored,1e-9)
    pm=pm+fnp.bincount(band,weights=restored-bgmu,minlength=fulln).astype(fnp.float32)
    pc_diag=fnp.diag(pc)
    pc=_fill(pc,pc_diag+fnp.bincount(band,weights=restored_var-pc_diag[band],minlength=fulln).astype(fnp.float32))
    return pm,pc

def _cat(arrays,axis):
    return arrays[0] if len(arrays)==1 else fnp.concatenate(arrays,axis=axis)

def _cols(a,seg,width):
    return a if seg[0]==0 and seg[1]==width else a[:,seg[0]:seg[1]]

def _assemble_blocks(P,rows_w,factors,none_order,q_order,xold,zcol,qcol,ystart,yend,fresh_pivots,big,qpair):
    """Column-pack the row projections of every history block (order: none_order then q_order).
    P is the projection of the packed parts (or None: per-block products as before)."""
    xs=[];ys=[];zs={};qs={}
    if P is not None:
        run=None
        for i in none_order:
            a,old,fr=xold[i]
            if old:run=(a,a+old) if run is None else (run[0],a+old)
            if fr:
                if run is not None:xs.append(P[:,run[0]:run[1]]);run=None
                xs.append(rows_w[:,fresh_pivots])
        if run is not None:xs.append(P[:,run[0]:run[1]])
        if yend>ystart:ys.append(P[:,ystart:yend])
        for i in none_order+q_order:
            a,r=zcol[i];zs[i]=P[:,a:a+r]
        for i in q_order:
            a,r=qcol[i];qs[i]=P[:,a:a+r]
    else:
        for i in none_order:
            f=factors[i];_,old,fr=xold[i]
            parts=fnp.concatenate((f[1][:,:old] if fr else f[1],f[2]),axis=1)
            both=matmul(rows_w,parts,depth=4) if big else rows_w @ parts
            xo,y=both[:,:old],both[:,old:]
            xs.append(fnp.concatenate((xo,rows_w[:,fresh_pivots]),axis=1) if fr else xo);ys.append(y)
        for i in none_order+q_order:zs[i]=rows_w @ factors[i][6]
        for i in q_order:qs[i]=rows_w @ factors[i][0]
    for i in q_order:
        x,y=qpair(qs[i],factors[i][1],factors[i][2])
        xs.append(x);ys.append(y)
    return _cat(xs,1),_cat(ys,1),zs,qs

def symmetric_covariance_product(w,cw):
    n=w.shape[1]
    if n<256 or n%64 or w.shape[0]%16:
        result=matmul(w.T,cw,depth=5,upgrade=False)
        return (result+result.T)*0.5
    groups=[]
    size=n
    for level in range(6):
        size=size//2
        groups.append((size,5-level,False))
    groups.append((size,6-6,True))
    if any((n//size if diagonal else n//(2*size))*max(size*w.shape[0],size*size)>26000000 for size,depth,diagonal in groups):
        result=matmul(w.T,cw,depth=5,upgrade=False)
        return (result+result.T)*0.5
    products=[]
    wt=w.T
    for size,depth,diagonal in groups:
        left=fnp.split(wt,n//size,axis=0)
        right=fnp.split(cw,n//size,axis=1)
        if not diagonal:
            left=left[::2]
            right=right[1::2]
        products.append(_batched(fnp.stack(left),fnp.stack(right),depth))
    diagonal=products[-1]
    diagonal=(diagonal+fnp.transpose(diagonal,(0,2,1)))*0.5
    for off in reversed(products[:-1]):
        diagonal=fnp.block([[diagonal[::2],off],[fnp.transpose(off,(0,2,1)),diagonal[1::2]]])
    return diagonal[0]

def aged_tensor_energy(x,y,ac,bc,cc,z,vv):
    extra=z @ vv.T
    r=y*bc+x*cc+extra
    xx=fnp.einsum('ij,ij->j',x,x);yy=fnp.einsum('ij,ij->j',y,y)
    rr=fnp.einsum('ij,ij->j',r,r);xy=fnp.einsum('ij,ij->j',x,y)
    xr=fnp.einsum('ij,ij->j',x,r);yr=fnp.einsum('ij,ij->j',y,r)
    energy=(ac*ac*(xx*yy*yy+2*xy*xy*yy)+xx*xx*rr+2*xx*xr*xr+2*ac*(xr*xy*xy+2*xx*xy*yr))/3
    return fnp.maximum(energy,0)

def aged_retained_columns(x,y,ac,bc,cc,z,vv,tolerance):
    energy=aged_tensor_energy(x,y,ac,bc,cc,z,vv)
    order=fnp.argsort(energy)
    return fnp.sort(order[fnp.cumsum(energy[order])>=tolerance*fnp.sum(energy)])

def tensor_history_diagonal(x,y,ac,bc,cc,z,vv):
    xy=x*y
    x2=x*x
    return fnp.einsum('ij,ij,j->i',xy,y,ac)+fnp.einsum('ij,ij,j->i',xy,x,bc)+fnp.einsum('ij,ij,j->i',x2,x,cc)+fnp.einsum('ij,ij->i',z @ vv.T,x2)

def reduced_history_repair(q,xr,yr,x,y,ac,bc,cc,z,vv,coeffs,mode):
    if mode==0:
        return fourth_traces(xr,yr,trace_weights(coeffs)),None
    ca,cb=paired_projection(q,xr,yr)
    traces=fourth_traces(ca,cb,trace_weights(coeffs))
    delta=tensor_history_diagonal(x,y,ac,bc,cc,z,vv)-tensor_history_diagonal(ca,cb,ac,bc,cc,z,vv)
    return traces,fnp.diag(delta)


def full_history_trace(x,y,ac,bc,cc,z,vv):
    xx=fnp.einsum('ij,ij->j',x,x);yy=fnp.einsum('ij,ij->j',y,y);xy=fnp.einsum('ij,ij->j',x,y)
    xr=fnp.einsum('ij,ij->j',x,z @ vv.T)
    left=ac*yy+2*bc*xy+3*cc*xx+2*xr
    right=2*ac*xy+bc*xx
    return (x @ left+y @ right+z @ (vv.T @ xx))/3

def projected_history_trace(q,x,y,ac,bc,cc,z,vv):
    xx=fnp.einsum('ij,ij->j',x,x);yy=fnp.einsum('ij,ij->j',y,y);xy=fnp.einsum('ij,ij->j',x,y)
    xr=fnp.einsum('ij,ij->j',x,(q.T @ z) @ vv.T)
    left=ac*yy+2*bc*xy+3*cc*xx+2*xr
    right=2*ac*xy+bc*xx
    return (q @ (x @ left+y @ right)+z @ (vv.T @ xx))/3

def isotropic_trace_mixed_slice(delta):
    n=delta.shape[0]
    return (fnp.ones((n,1),dtype=fnp.float32)*delta+2*fnp.diag(delta))/(n+2)

def trace_summary_history_repair(q,xr,yr,x,y,ac,bc,cc,z,vv,coeffs,strength):
    traces=fourth_traces(xr,yr,trace_weights(coeffs))
    if not strength:return traces,None
    delta=full_history_trace(x,y,ac,bc,cc,z,vv)-projected_history_trace(q,xr,yr,ac,bc,cc,z,vv)
    return traces,isotropic_trace_mixed_slice(delta)*strength


def correlation_mixed_third(c, t, a2, a3, a4, b2, b3, b4, order):
    result = .5*c*(t*fnp.outer(a3, b2)+t.T*fnp.outer(a2, b3))
    if order >= 2:
        result = result+.25*c*c*(t*fnp.outer(a4, b3)+t.T*fnp.outer(a3, b4))
    return result


def gaussian_third_star_shift(c, gaussian_g2, g3_over_g2):
    shift = .5*(gaussian_g2[:, None]*(c*c))*g3_over_g2
    return shift-fnp.diag(fnp.diag(shift))


def fused_gaussian_star_closures(c, t, g2, g3, q2, q3, c_squared=None):
    left3 = fnp.stack((g3, q3, q3))
    right2 = fnp.stack((g2, g2, q2))
    left2 = fnp.stack((g2, q2, q2))
    right3 = fnp.stack((g3, g3, q3))
    # mixed[k] = .5*c*(t*outer(left3[k],right2[k]) + t.T*outer(left2[k],right3[k])), contracted in place.
    mixed = fnp.einsum('ij,ki,kj->kij', c*t, .5*left3, right2)+fnp.einsum('ij,ki,kj->kij', c*t.T, .5*left2, right3)
    gaussian_c3 = (c*c if c_squared is None else c_squared)*c/6
    cubic = fnp.einsum('ij,ki,kj->kij', gaussian_c3, left3, right3)
    return mixed, cubic


def _uu(x,z,z2,a1,a2):
    if a1==0.0 and a2==0.0:return x
    u=1.0+a1*z+a2*z2
    return x*fnp.outer(u,u)

def _sc(x,s):
    return x if s==1.0 else x*s

class Estimator(BaseEstimator):
    mean_translation_fraction=1.0586134663393054
    diagonal_transport_enabled=True
    diagonal_transport_beta=[-0.018201120827605523, -0.06776831109092474, -0.050638663341394295, -0.10280807738824169, 0.03399773964931804, 0.0042077385238525, -1.963318094740562e-06, 7.032106478314336e-07, -0.01837831898424286, 0.03057850843236052, -0.029606143231891372, -0.05900841726032715, 0.06142319646910776, -0.02664788187232977, -0.039861142280003634, 0.025990541709657082, 0.06218172448561374, 0.06748410028814217, -0.30530579962978094, -0.022242643299185644, 17.697683994662036, -3.250177776954324, 0.06696240362205018, 20.71827360033118, 8.588358271251217, -18.4800849137816, -1.850251142973923, -21.962551793902087, 0.29248874058760443, -0.5580531155072045, 0.0933017857254042, -18.7715723841847, -7.092257274895998, -16.006098568270936, 4.434326316755105, 0.11778284222522067, 2.944231368457651, -5.8488611513312465, 1.7926797040773654, -1.087141287053094, -8.476498136769273, 21.82509895435944, 0.028473284747354444, 0.44685325796318337, -0.5103087811743914, -5.591137450642672, -0.2723721794996479, 2.4311201948587144, -0.1408762186400088, -12.617445777837288, -1.9699954456082793, -6.014972738408559, 0.0014129065839292505, 0.03202929395030387, 0.022185557003173248, 0.19925014288579312, -0.006304809375119503, 0.0038224323902354917, 0.027529111021499828, -0.1787749475296705, -0.04444599280775076, -0.09819338304752058, -9.80898773681621e-06, 1.3950811137370848e-05, 0.052340838711287654, 0.05689315014930499, 0.21452619040238385, -0.011856981409675263, -0.009282048501266201, -0.02785529022831245, 0.03816928724778866, -0.0036329595281177386, -0.049476938853371635, -0.07427657554804022, 0.013887954438945499, 0.059173356964898546, 8.115783405901027, 1.7426240964987894, 0.1814503315387423, 16.324947117833222, 2.4821026480587847, -9.151301469949587, -3.88547280015699, 1.3752280926747118, -0.02580698837929274, 4.217073318973226, -0.7757437966715915, 7.651702635913064, 1.4895732598510165, -1.1391498605779373, 0.31516247539690756, -0.012559695989973642, -13.3908131810193, 9.60307869132944, -0.7157021105512896, -1.6783177179057163, -7.058747943414089, -1.9107671294374657, 0.2176802822664925, 2.3973481521032807, 0.0014107565939727065, 9.251839848369197, 0.04317702131450774, -3.9383357706150064, 0.04995453957695896, -8.75896920628923, -5.61933697894971, 1.7615921717587788, -0.16337762017994825, 0.009008742926161748, 0.029205931230450122, -0.09491504011321832]
    shadow_hidden_modulation=[0.09161783316520612, -1.1680906175829529, 0.2588504147506245, -3.0552570214213848, 3.0006433833034225, -1.1174297688530648, 3.5908729443172978, 0.3025251686780723, 0.1559415736856713, 1.3217488017848766, -1.2806205560558492, -1.3966382239980155, 2.293581634958678, -0.2124499161601669, -4.618138219752177, 0.17118872767402823, -0.776301199294486, 1.9053936040335235, 1.6422075483333967, 1.2520793195856499, 2.667190058551322, 1.3694167106891464, 0.52840981816506, 3.6763178033480832, -0.19711334486351048, -1.9206397987390815, 0.09836468853404488, -2.6041873335873804, 0.48969167715312006, -0.964200760652054, 0.8139171186881069, -1.7631794970221553]
    tree_hidden_modulation=[-0.6955495854948561, 0.07346271602939893, 0.24375314143696317, 3.0403275029836037, -1.56503444252845, 1.8544739727088442, -2.625229119739274, -2.0286268667058813, 0.16453494141256444, -1.1571895012215034, 0.6086704753196661, -0.7248928249056189, -2.1304079214255927, 3.6922350203511596, 4.133031908043832, 0.16662663863960683, 0.8126379127336619, -1.6641193181133418, 0.4474941364852664, -4.4844071230565685, -2.5827489540454764, -0.06859326127789069, 0.13919611751227207, 0.6909588990280972, -0.31416665361117974, 1.6039593916531938, 0.01668408872286021, -0.10687684734215964, 0.4693156245667823, 1.8000733342984492, 0.5110692652275186, 2.173614511483405]
    fourth_modulation=True
    shadow_modulation=[-0.06161776503489524, 0.33326090209148806, -0.16912963525638464, -0.12199216203473935, 0.0405753039560021, -0.010454634498283516, -0.0646892831769453, -0.003993770564717825, 0.01536248114324788, 0.09926047960287707, -0.02018319383774815, 0.03595283036960493, -0.05151635107782654, 0.15954260455412303, -0.1213412743950661, -0.24614396513212966, -0.289565638123371, 0.25188737712375303, 0.1261442856615268, -0.0389568767395267, -8.050897060280643e-05]
    tree_modulation=[-0.4023895632021009, 0.1305351767296476, -0.16564584983785596, -0.09083067889709308, -0.0024360085994233893, 0.040646179778005755, -0.07634036272018048, -0.026080978628957738, 0.10032337075765799, 0.07703133642497212, -0.017493712274795674, -0.012337098092943068, -0.010074220345548694, 0.09932689650126575, -0.06441909253130568, -0.3397663214442417, 0.6116050392466502, -0.3211982917847173, -0.7673635364033105, 0.09617028568322478, 0.28992835843645276]
    radial_adjustments=[0.06920568423465163, 0.02015321882826334, 0.0869614457479891, 0.04896280676827293, -0.024131699183302435]
    birth_energy_tolerance=0.0001
    aged_energy_tolerance=0.0005
    aged_every_age=False
    calibrated_mixed_depth=[0.8707821579338159, 0.8722540275619666, 0.8679386582012307, 0.8636232888404951, 0.8593079194797593, 0.8549925501190235, 0.8506771807582878, 0.846361811397552, 0.8420464420368163, 0.8377310726760805, 0.8334157033153448, 0.829100333954609, 0.8247849645938732, 0.8204695952331376, 0.8161542258724017, 0.8176260955005524]
    tail_repair_mode=0
    tail_repair_start=0
    trace_summary_strength=1.5010018073188356
    correlation_third_order=1
    correlation_third_all=True
    fused_star_closures=True
    gaussian_star_strength=1.0001221014186057
    gaussian_star_pairs=True
    repair_rank=2
    tail_rank=256
    shadow_weight=0.7349710623372364
    tree_weight=0.6828535927929209
    compress_age=5
    cutoff=2.35
    band_limit=3.0
    kink_cutoff=3.0
    readout_cutoff=2.5
    cr24_beta=(0.0004394782096245699,0.0006782778361671054,-0.0013704376246224388,0.002771035435229722,-0.0009908664465131986,0.003142898914146825,0.0011467307463203097,0.006780475886009378,-0.0017627143530005065,-0.004624435859810974,0.0014755210338954104,-0.0020878758588302447,-0.00964091617670603,-0.00038169844925047073,-0.00046284723881967384,0.002384279268030121,0.0016489215178798677,0.001331705649946801,0.0010599932215796834,0.0002634029183784632,-0.0005545391255218395,0.021078433754140393,3.302888845283773e-05,0.00579992645003612,0.008041500683216709,0.025358285342016208,-0.002442004049698083,0.010464788495593779,-0.002720521861950897,-0.0018573066834094679,0.015328000524770123,-0.0032223666354788457,-0.014534871323809376,-0.0005891022411489886,-0.0007143449134619483,0.007526244967357396,0.013114273156526949,0.019579038707334816,0.023809467500413396,0.00040652837297772685,0.004978072948835551,0.003030812020159297,-0.0028058351839655563,0.0019002734481653517,-0.006585797630677938,-0.0031034181453900443,-0.02114375426021418,0.005496723200173734,0.00323296045921769,-0.001241347603370428,0.006510683737716009,0.00033692057686460203,0.0011902613250370853,0.0014433099449302777,-0.002233899950485422,-0.0007473202264052936,0.0006539902155944254,0.0022979806499678055,-0.0008213769462872967,-0.0018637559569086122,0.002208894953530524,-0.008205506022778478,0.0029725691650111007,-0.0010839694597498414,0.04275283401171834,-0.011114416659077203,0.00233950281566933,-0.008378731159273784,-0.013164652677828697,-0.017648509649475615,-0.0024067175693387576,-0.002918383828320525,0.000720921855444785,-0.0016507802368460136,-0.003509848760350194,-0.005600653768801219,0.0016608305135158371,0.00853965905089886,-0.004650845820193992,-0.013715544241014264,-0.005819029897225951,-0.015287552146388524,0.00397429148687051,0.018991925310901048,-0.002405784381103404,0.004707414583234565,0.02780343766193995,0.0008605937171979655,0.0010435552634103767,0.0024187154533942155,0.0027995588859264582,0.002012057856519305,0.0014148531483351395,-0.0005938795326347454,-0.003410997152638582,0.0024505835939613407,0.001384782257805567,0.04849011812835448,-0.01260593336522579,0.0036176794971180715,-0.009040572669954994,-0.014931304046217968,-0.00375565885642782,-0.0027296908365608105,-0.0033100209576829385,0.0187583130219859,0.022208666545808955,0.01969952837444966,0.013709415436691378,0.0018837082886589524,0.00636037210484172,0.003911027658324984,0.01769229964736762,-0.004599451579847128,-0.0005960931700881397,-0.003627036295512693,-0.005447895684900976,0.0014046667885885139,-0.0009959659841886877,-0.0012077075677033096,0.002008654771863435,0.0016591363031323374,0.000412762698401213,-0.0013263680965110507,0.0006872973871291075,0.0017890994958206475,0.10461236127896636,-0.027195983561159626,-0.07134796480721979,0.022765030371297113,-0.03221272773881463,-0.14874457532497581,-0.005889022650474015,-0.007141024225982583,0.03678577851720407,0.025440334344333154,0.020546179193798135,0.01635407244515262,0.004063903732327379,-0.00855568964763473,0.0070701159289075334,0.018548267664016488,-0.0059182049249351335,0.00837431450079912,0.03866899643391546,0.0015309640393466948,0.0018564457878627678,-0.009563166489891185,-0.00661370134601721,-0.005341372135688728,-0.004251553830985959,-0.0010564894792955738,0.0022242151133980057,-0.007111076681154231,0.006817006016659634,0.021969799141256453,-0.02272316673317754,0.004016444860498227,0.004870337873272505,-0.007514536097055137,-0.0034467008674532924,0.0047108224615662235,0.01439592400995978,-0.0027716730309995286,-0.01494023281148747,0.0016217440290297272,-0.007009914663345058,-0.03287222535740048,-0.0012815290454454996,-0.001553981110786787,0.011025584923299284,0.017503261672145647,0.01701205904496039,0.01295167157938418,0.000884359082988273,0.004834894706912895,0.009919093839979456,0.04580212556967137,0.0018133754076776725,0.00219889759058565,-0.011327249031719916,-0.007833706779747237,-0.006326675627362274,-0.005035822503576507,-0.0012513762511640663,0.0026345079860575827,-0.026223968386208284,0.008373390701544765,0.010153566967275176,-0.03585506878875701,-0.034126775450031725,-0.017819667141272373,0.006665277698229868,-0.005778319382333744,-0.026890402380940693,0.00033151519909164784,0.0004019950692169974,-0.002070809608431687,-0.0014321319522255865,-0.0011566215805714243,-0.0009206321497461613,-0.000228772401614665,0.00048163189806640314,0.00048745890419912434,-0.002511062099587223,-0.0017366020768879282,-0.0014025184172966694,-0.0011163578194124525,-0.00027740923394731497,0.0005840264601157595,0.0016432086000484028,0.006050464747804235,0.01070717615166617,0.012843506205919859,0.0014290267496198039,0.009422342728427726,0.015529644287007892,0.02318402760278774,0.026741955424271577,0.000988287315445463,0.008833430059965737,0.03137462161991362,0.03432214623896124,0.0007981627915454711,0.009616367595943788,0.036070659382090794,0.0006353109253446607,0.010953681736592544,0.00015787153012561321,-0.00033236511121275937,-0.006413635953116031,0.7550647674062554,0.04830046354329554,0.02823640647379145,-0.05193604611675345,-0.44905765736007885,-0.11061124754729407,-0.5048352031101183,0.06706529610291803,0.002169696856153159,0.4885592023581113,0.5782360695363077,0.5817552999754441,-0.14344668506663308,-0.39568601331801223,0.17032449142352396,-0.0007860405038321447,-0.13320780449772862,-0.18743815214817572,-0.5777398262907767,-0.0005282271089264185,0.10096164204206219,-0.009349715626023918,0.0006488042850266064,-0.07226586024442337,-0.0015796523351746456,0.1347032898409989,-0.0004719394642689173,0.5523738153029573,-0.005302548363110123,-0.10346141973344287,-0.7968149003795173,-0.1050106948938578,0.0020212703662405673,0.0013759439554623806,-0.0034366844891405425,-0.00024450988515227354,-0.0012811962692976134,-0.0006548125853051919,-0.00012368010926480693,0.031185106968329945,-0.00810716483283599,0.009714597254217071,0.0051439965980407615,-0.00960266404436362,0.007369567663296987,-0.001755526775647582,-0.002128750384966026,0.00476636230954206,0.006631242856961385,0.005104877428776795,0.0007440600363066709,0.001211455998590976,0.00455490536603478,0.5990963699124953,0.05152228227386865,-0.004719481145198049,-0.08979230734007597,-1.0250096123207146,-0.6647298438412306,-0.05042360085468509,0.020970190063004313,-0.0002759975076008461,-0.014436815047042337,0.004227627358147336,0.13406912355747028,0.1342562024800514,-0.6468220064517404,0.11157592347784352,-0.0012760695367906307,-0.14259748747698142,0.06985660498504491,-0.25968596649870807,-0.07409661089238605,-0.05790401493055258,-0.07872847326049152,-0.0046454483779824265,-0.44201936984145973,-0.00541667772716668,0.1552101986402565,0.00038600545311233696,0.3024179636940218,0.0007579022629364692,-0.37377368408191236,-0.5762368566320608,-0.3222321743228994,0.0009500795419461517,0.002726337250224124,0.0008659018214888863,0.0037387329063926257,0.0003981746816780594,-0.000652201129381859,0.0017682700075848704,0.01465827265806608,-0.003810698251708775,-0.0021283335396324867,-0.0031248222314583427,-0.00451364389896243,0.005583434723349619,-0.0008251692117685397,-0.0010005995360376485,-0.003049617424677937,-0.007839351722879615,-0.01018583013985911,-0.010403971209266148,0.0005694337479305398,0.003899141299428348,-0.4486245485314778,0.03414016932296395,-0.0026652400327882007,0.0989072202840221,0.15458022653165784,0.47519139509750896,0.21113838939622662,-0.12021400332851301,-0.004362621765262566,-0.4565698151843425,-0.2550533971445061,-0.07466618352981566,-0.14231317612920505,0.6023997234829614,0.0220054931462555,0.005713463047756682,0.013570633764671285,-0.11992011269638514,-0.2171048481392637,-0.02766301071496031,-0.06827303432469305,-0.015515652994759061,0.0120800862451393,0.11540102878882276,-0.0007596528660084844,0.25476361388925256,0.003388601346594104,-0.10052258733904336,-0.004070104110151098,0.23087821932484423,0.45920487157575973,0.2723804662777608,-0.007184033001251185,-0.0017757689151202513,-0.016126681760594064,0.006355704202747639,0.0005087800587459851,0.010526689677720396,-0.0021137745849052806,-0.010510512474032934,-0.011368521701345101,-0.0008805164944316657,-0.008997895772511751,0.0012626951798310213,0.00383856911033256,-0.0021188340742404614,0.007516292520759174,-0.0014400596731112916,0.008294530516440355,0.000940125831728697,0.0073800538154829235,0.007026796099760685,0.004168683421896319,-0.002273988276498209,-0.00109471441918159,-0.0021314895315385743,-0.08177827932426583,-0.004824961093329254,-0.06034225568625135,0.00834614126445991,0.00674672317229251,0.017300933457558478,-0.04649820255484153,-0.009522191492691023,-0.016777263104293436,0.01569987049227106,-0.08087838898968898,-0.008866044586734107,-0.03561467341575934,-0.006574202980810272,0.020678905337289694,-0.0040239276879530834,0.022238882573121076,0.0116360395013898,0.013008693882687586,0.005155051651831463,-0.00155005665059585,-0.006452225243135383,-0.03439245171383551,-0.005748090484195762,-0.00324799154573987,0.0023620608651212615,0.004265798335173199,-0.004167848659918316,0.012033078270189334,-0.008756202174262233,0.008853563311573241,0.005251196142922692,0.0017862645697578488,-0.0009101036826232032,-0.0014814141506037809,-3.037218897065953e-05,-0.0038104382657579935,0.003358450975425734,-0.003915666421597712,0.0027670763681707966,0.000778230712125772,0.0003475652002684087,-0.004551524656296327,-0.005267645874904144,0.0013849266033368805,0.0036208810368388335,0.0006888048716562934,0.003380418908865205,)
    s_mu_td=0.9990258631518936
    s_mu_kd=1.016197775279754
    s_var_td=0.9542355936773979
    s_var_kd=0.8817565551260007
    s_pg_td=1.0950852634332835
    s_pair_k22=0.9955209569650958
    s_corr=0.9924885553583811
    s_fmix=1.1359770675306888
    s_fcub=1.1670394588165887
    a_pair1=0.01532970431022202
    a_pair2=0.05236793575240955
    a_corr1=-0.00045733375995361494
    a_corr2=-0.0024656687966504607
    a_fmix1=0.11951855676692123
    a_fmix2=0.05730090677580626
    a_fcub1=0.07363536127126542
    a_fcub2=-0.009907418994523739
    s_fhigh=0.8526123993311298
    def setup(self,ctx):
        flops.configure(symmetry_warnings=False)

    def predict(self,mlp,budget):
        n=mlp.width
        fulln=n
        weights=[fnp.asarray(w) for w in mlp.weights]
        _cr24={}
        _cr24_extra=None
        pm=fnp.zeros(n,dtype=fnp.float32)
        pc=fnp.eye(n,dtype=fnp.float32)
        eye=fnp.eye(n,dtype=fnp.float32)
        previous=fnp.arange(n)
        mu=fnp.zeros(n,dtype=fnp.float32)
        cov=eye
        radial=fnp.asarray(-8.012998596100438/(fulln+2),dtype=fnp.float32)
        factors=[]
        fresh_pivots=[]
        rows=[]
        # Mean, shadow and tree response channels are carried as one (3, fulln) state.
        channel_mean=fnp.zeros((3,fulln),dtype=fnp.float32)
        channel_history=[]
        channel_weights=fnp.asarray([1.0,self.shadow_weight,self.tree_weight],dtype=fnp.float32)
        channel_offsets=fnp.asarray([[0],[fulln],[2*fulln]])
        radial3c=fnp.zeros((2,1),dtype=fnp.float32)
        coef_scale=fnp.asarray([[2],[1],[2],[3],[1],[2],[1],[1],[0],[0],[0],[1],[1/3],[2/3],[1],[1]],dtype=fnp.float32)
        qcache={}
        def qpair(q_,xr_,yr_):
            # paired_projection with the (xr|yr) concatenation of a compressed block cached across layers.
            hit=qcache.get(id(xr_))
            if hit is None or hit[0] is not xr_:
                hit=[xr_,fnp.concatenate((xr_,yr_),axis=1),None];qcache[id(xr_)]=hit
            both=compact_product(q_,hit[1])
            return both[:,:xr_.shape[1]],both[:,xr_.shape[1]:]
        def qryx(xr_,yr_):
            hit=qcache.get(id(xr_))
            if hit is None or hit[0] is not xr_:
                hit=[xr_,fnp.concatenate((xr_,yr_),axis=1),None];qcache[id(xr_)]=hit
            if hit[2] is None:hit[2]=fnp.concatenate((yr_.T,xr_.T),axis=0)
            return hit[2]
        # Feature columns are numerator * norm^-pinv * sd^-psd * const (see the original ratios).
        local_pinv=fnp.asarray([0,0,0,0,1,2,0,0,2,3,2,0,1,3,0,1,1,2,0,0],dtype=fnp.float32)
        local_psd=fnp.asarray([0,0,0,0,1,0,0,0,0,0,1,1,0,0,1,0,1,0,0,0],dtype=fnp.float32)
        local_const=fnp.asarray([1,1,1,1,fulln,fulln,0,0,1,1,fulln,1,fulln,fulln*fulln,1,1,1,1,1,fulln],dtype=fnp.float32)
        # Closure weights 1+a1*z+a2*z2 for the active (a1,a2) pairs, evaluated together per layer.
        uu_pairs=((self.a_pair1,self.a_pair2),(self.a_corr1,self.a_corr2),(self.a_fmix1,self.a_fmix2),(self.a_fcub1,self.a_fcub2))
        uu_index={}
        for k_,(a1_,a2_) in enumerate(uu_pairs):
            if not (a1_==0.0 and a2_==0.0):uu_index[k_]=len(uu_index)
        if uu_index:
            # 1+a1*z+a2*(z*z-1/3) = (1-a2/3)+z*(a1+a2*z)
            uu_a1=fnp.asarray([[uu_pairs[k_][0]] for k_ in uu_index],dtype=fnp.float32)
            uu_a2=fnp.asarray([[uu_pairs[k_][1]] for k_ in uu_index],dtype=fnp.float32)
            uu_c0=fnp.asarray([[1-uu_pairs[k_][1]/3] for k_ in uu_index],dtype=fnp.float32)
        feature_mean=fnp.asarray([0.40502333641052246, 0.7080164551734924, 0.5533769130706787, 0.21758921444416046, 6.896132469177246, 10.278403282165527, -0.17221391201019287, 2.886648178100586, 7.2637248039245605, 1.0476505756378174, 101.9793472290039, 4.1848249435424805, 6691.6767578125, 180548.765625, 1.0070170164108276, 3.119014024734497, 12.911785125732422, 64.68301391601562, 0.4511948525905609, -0.6836428046226501],dtype=fnp.float32)
        feature_scale=fnp.asarray([1.8177217245101929, 0.29656505584716797, 0.3460083305835724, 0.14828214049339294, 38.397953033447266, 17.37635612487793, 2.656996965408325, 11.578130722045898, 3.925035238265991, 0.2454264611005783, 484.484130859375, 27.490198135375977, 5811.8984375, 190149.359375, 1.2785800695419312, 6.5786943435668945, 40.7035026550293, 286.3301086425781, 0.2914688289165497, 2.0810325145721436],dtype=fnp.float32)
        feature_linear=fnp.asarray([0.589740612022238, -1.4697368983899024, -0.41205109362845416, 0.4474101637836371, 1.5972671832455192, -1.3527812307955496, 0.03691162990406692, -0.07288373281553058, 0.22015477657694962, 0.028330763209275665, -0.4033521511427395, 0.1404456086864079, -0.4456829307743688, -0.8579877593317814, -0.12309511627953039, 0.1537531786410473, 0.01808732201762056, -0.1336267087947339, 0.07131326139579658, -0.0769012657487013],dtype=fnp.float32)
        feature_quadratic=fnp.asarray([[-0.9864887213628191, 0.8225589340330962, -0.362914106536392, 0.19046995533611383, 1.0952118809715148, 0.7404364890896833, -0.17509024256869782, -0.025968662043054924, 0.379262498586167, -0.08603386061142536, -0.8195603552153607, 0.72365705120791, 0.09122860240637617, -0.9961746584387361, 0.22062641849215145, -0.019625762914761914, -0.2975788127451582, -0.3855884935913855, 0.42536136272967806, -0.261733193257597], [0.8225589340330962, -0.22622678494038356, -0.09312419380264542, -0.059468327123746174, 0.03518905452202201, -0.033048112973744784, -0.00861399589112527, 0.17366391152738705, -0.008649113110415383, 0.0055201128838520495, -0.015692260352408054, 0.026226488971319926, -0.03582522468179988, -0.19337505079724676, -0.4848181710031167, -0.8657171902460398, -0.9038064245898008, -0.6678918582998584, -0.09282678781499515, -0.018702116001736824], [-0.362914106536392, -0.09312419380264542, -0.3352337767730864, -0.10246048766008584, -0.11950662378055123, -0.34473439511154014, 0.006068784651209476, 0.046610749059916134, -0.1170173390089966, 0.22756020335511773, 0.2406073952034895, -0.07944346183320188, -0.18621778167110808, -0.06732382833683884, 0.46339555467531535, 0.2208469362306012, 0.07772689447545161, -0.030075254144907027, 0.11231464643264674, 0.11880916108667866], [0.19046995533611383, -0.059468327123746174, -0.10246048766008584, 0.04497071703523024, -0.13644694860977433, 0.26541338115208923, 0.014144768346187964, -0.05568707419019702, -0.0355488138273476, -0.04799660128323635, 0.08203779278136535, -0.07472561935154429, 0.08018264555224151, 0.08978821518468462, 0.5611296600978541, 0.03747798319116332, -0.0779255104057069, 0.008051174642993864, -0.23457792460235619, -0.150401830975232], [1.0952118809715148, 0.03518905452202201, -0.11950662378055123, -0.13644694860977433, -0.04707213503815906, 0.004298907468007538, 0.10644129482486303, -0.21250171311762422, 0.009044598219507918, 0.004485957654965681, 0.03109181908469828, -0.019893670510031554, -0.014652707116269027, 0.015931945441596214, -1.7989276226626976, 0.36016778288412216, 2.1594069992833638, 2.62228385206325, 0.013116179055083103, 0.0076312141514691435], [0.7404364890896833, -0.033048112973744784, -0.34473439511154014, 0.26541338115208923, 0.004298907468007538, 0.04418084134093953, 0.0010725752368633684, 0.1576406423513807, -0.05001526270253658, -0.003159019616194727, -0.00177263789509772, 0.029244771128524725, -0.013755297594131816, 0.012335022020738447, 0.5434828538592573, -1.381105076323018, -1.3122385770506615, -0.7877811691123727, -0.0439172685608305, 0.08437777143571298], [-0.17509024256869782, -0.00861399589112527, 0.006068784651209476, 0.014144768346187964, 0.10644129482486303, 0.0010725752368633684, 0.298391608813894, -0.04083731084572562, -0.028046514342726606, -0.03165324854797744, 0.09406320888893217, 0.016557415356895873, -0.055163320253324453, -0.008877979895953483, -0.0737676718254968, -0.05305415126477859, -0.059575132063758515, -0.07049088208691387, 0.01569259971651469, -0.0009487389415525462], [-0.025968662043054924, 0.17366391152738705, 0.046610749059916134, -0.05568707419019702, -0.21250171311762422, 0.1576406423513807, -0.04083731084572562, 0.037011738833260185, -0.01883068431500281, 0.004443010122792043, 0.024050988698223724, -0.020447382061585702, 0.0655283501472437, 0.10232219634529943, 0.03242939843828993, -0.004957158294006968, 0.01247501828002073, 0.032856410931723395, -0.012166155240706198, 0.009208584368115651], [0.379262498586167, -0.008649113110415383, -0.1170173390089966, -0.0355488138273476, 0.009044598219507918, -0.05001526270253658, -0.028046514342726606, -0.01883068431500281, -0.010155130178633887, 0.001280719216497256, -0.0066329382711065345, 0.021854771784691664, 0.0022583004122862555, 0.03575393003486385, -0.08506729792096247, -0.49125237434101293, -0.12952839962701512, 0.06413716286200632, -0.04121092909016981, -0.01437851857171912], [-0.08603386061142536, 0.0055201128838520495, 0.22756020335511773, -0.04799660128323635, 0.004485957654965681, -0.003159019616194727, -0.03165324854797744, 0.004443010122792043, 0.001280719216497256, 0.0007296604333607781, -0.004067818453560862, -0.004286475810503194, -0.009922873382061138, 0.005575091717852387, -0.786909825054678, 0.8097690733962111, 0.2749113597736144, -0.33956194036925236, -0.0012784441169816294, 0.04540033893399136], [-0.8195603552153607, -0.015692260352408054, 0.2406073952034895, 0.08203779278136535, 0.03109181908469828, -0.00177263789509772, 0.09406320888893217, 0.024050988698223724, -0.0066329382711065345, -0.004067818453560862, 0.001023944877195539, 0.00606007910890211, 0.0043913291076778365, -0.012771087751254974, 0.5601052074190889, -0.03166125901043021, -0.5645971493923649, -0.33561497246469585, 0.0051077171086087066, 0.018161791856282986], [0.72365705120791, 0.026226488971319926, -0.07944346183320188, -0.07472561935154429, -0.019893670510031554, 0.029244771128524725, 0.016557415356895873, -0.020447382061585702, 0.021854771784691664, -0.004286475810503194, 0.00606007910890211, -0.026330106235887586, 0.01565991216503481, -0.01460901522137699, -1.0454630231714486, 0.8687052746307676, 0.28748057460821025, -0.3558453458808533, -0.007330071146457231, -0.10655688326822307], [0.09122860240637617, -0.03582522468179988, -0.18621778167110808, 0.08018264555224151, -0.014652707116269027, -0.013755297594131816, -0.055163320253324453, 0.0655283501472437, 0.0022583004122862555, -0.009922873382061138, 0.0043913291076778365, 0.01565991216503481, -0.029364901591114774, 0.010876162972959506, 0.6450717413078848, -0.6256693438664764, -0.5710259934768399, -0.18714836506998261, 0.11909707986389938, 0.1523925170479518], [-0.9961746584387361, -0.19337505079724676, -0.06732382833683884, 0.08978821518468462, 0.015931945441596214, 0.012335022020738447, -0.008877979895953483, 0.10232219634529943, 0.03575393003486385, 0.005575091717852387, -0.012771087751254974, -0.01460901522137699, 0.010876162972959506, -0.02412278865199241, 1.7164092685928984, -0.296986789565199, -0.21503046677559975, 0.2278036983391619, 0.07367818259267096, 0.34859045564779556], [0.22062641849215145, -0.4848181710031167, 0.46339555467531535, 0.5611296600978541, -1.7989276226626976, 0.5434828538592573, -0.0737676718254968, 0.03242939843828993, -0.08506729792096247, -0.786909825054678, 0.5601052074190889, -1.0454630231714486, 0.6450717413078848, 1.7164092685928984, 1.1545095314793885, 0.13041110972225606, -0.25807632512918405, -0.40195708940990516, -0.7096679984848382, 0.1930159825983251], [-0.019625762914761914, -0.8657171902460398, 0.2208469362306012, 0.03747798319116332, 0.36016778288412216, -1.381105076323018, -0.05305415126477859, -0.004957158294006968, -0.49125237434101293, 0.8097690733962111, -0.03166125901043021, 0.8687052746307676, -0.6256693438664764, -0.296986789565199, 0.13041110972225606, -0.7891276031918576, -0.6673944087972926, -0.6489180048392641, -0.6302599247643497, -0.29554099060493144], [-0.2975788127451582, -0.9038064245898008, 0.07772689447545161, -0.0779255104057069, 2.1594069992833638, -1.3122385770506615, -0.059575132063758515, 0.01247501828002073, -0.12952839962701512, 0.2749113597736144, -0.5645971493923649, 0.28748057460821025, -0.5710259934768399, -0.21503046677559975, -0.25807632512918405, -0.6673944087972926, -1.5131366105796469, -0.6259530766930281, 0.21493333392748998, -0.5569162720423979], [-0.3855884935913855, -0.6678918582998584, -0.030075254144907027, 0.008051174642993864, 2.62228385206325, -0.7877811691123727, -0.07049088208691387, 0.032856410931723395, 0.06413716286200632, -0.33956194036925236, -0.33561497246469585, -0.3558453458808533, -0.18714836506998261, 0.2278036983391619, -0.40195708940990516, -0.6489180048392641, -0.6259530766930281, -0.940291903256531, 0.5129304275614472, -0.5029407343194516], [0.42536136272967806, -0.09282678781499515, 0.11231464643264674, -0.23457792460235619, 0.013116179055083103, -0.0439172685608305, 0.01569259971651469, -0.012166155240706198, -0.04121092909016981, -0.0012784441169816294, 0.0051077171086087066, -0.007330071146457231, 0.11909707986389938, 0.07367818259267096, -0.7096679984848382, -0.6302599247643497, 0.21493333392748998, 0.5129304275614472, -0.19896700925996388, -0.25782798196231327], [-0.261733193257597, -0.018702116001736824, 0.11880916108667866, -0.150401830975232, 0.0076312141514691435, 0.08437777143571298, -0.0009487389415525462, 0.009208584368115651, -0.01437851857171912, 0.04540033893399136, 0.018161791856282986, -0.10655688326822307, 0.1523925170479518, 0.34859045564779556, 0.1930159825983251, -0.29554099060493144, -0.5569162720423979, -0.5029407343194516, -0.25782798196231327, -0.05914297901546737]],dtype=fnp.float32)
        feature_bias=0.27349081605904907
        neural_0=fnp.asarray([[0.23071880638599396, 0.04557029902935028, 0.00188479817006737, -0.003140551969408989, -0.04175648093223572, 0.0036893729120492935, 0.5184378027915955, 0.012306030839681625, -0.3529600501060486, 0.14674168825149536, -0.13996237516403198, 0.09135648608207703, -0.006626873277127743, 0.04496999830007553, 0.004352435935288668, 0.01731123775243759, 0.056409355252981186, 0.056069619953632355, 0.20215056836605072, -0.10950672626495361, 0.06482002139091492, -0.0044833943247795105, -0.2173968106508255, 0.005883856676518917, 0.009735377505421638, 0.10568157583475113, 0.0042268685065209866, 0.041801754385232925, 0.45603278279304504, 0.07479993253946304, -0.13448917865753174, 0.09767676144838333], [0.2830241024494171, -0.22981303930282593, -0.25075826048851013, -0.05013277009129524, 0.12385451793670654, 0.04413124918937683, -0.6075645089149475, 0.006112684961408377, -0.34132689237594604, -0.41747015714645386, 0.006320552434772253, -0.16080684959888458, 0.0010103932581841946, 0.15069907903671265, 0.037399280816316605, 0.46974655985832214, -0.13495035469532013, -0.25542309880256653, -0.3190755248069763, 0.14776603877544403, 0.4566870927810669, -0.0172394048422575, 0.49527719616889954, 0.016809780150651932, 0.05876229330897331, 0.7145564556121826, 0.038238275796175, 0.2416539490222931, -0.28608208894729614, -0.1447792649269104, 0.2928032875061035, -0.5464407801628113], [0.12470594048500061, -0.11985784769058228, 0.013101859949529171, 0.005018648225814104, 0.09261389076709747, -0.003986048046499491, 0.5397156476974487, 0.0024213972501456738, -0.674014151096344, 0.008441600017249584, -0.29943981766700745, -0.1331731081008911, 0.01708528771996498, 0.04969526082277298, -0.0028103867080062628, 0.056753747165203094, -0.020800665020942688, -0.03294410929083824, 0.013759706169366837, 0.0895807072520256, 0.039542436599731445, 0.427561491727829, 0.06607929617166519, 0.0021042099688202143, -0.08192777633666992, -0.08197667449712753, -0.0028474319260567427, 0.12733164429664612, 0.1971508413553238, -0.045145921409130096, 0.08223146945238113, 0.008861701935529709], [0.4609464704990387, 0.1345192939043045, -0.2896898090839386, 0.0003668717690743506, 0.12809999287128448, -0.00022366701159626245, 0.16232454776763916, -0.2987738251686096, 0.21758811175823212, 1.0042603015899658, -0.3576202094554901, -0.060756683349609375, -0.026532765477895737, 0.5148457288742065, -0.00012100275489501655, -0.019413592293858528, -0.0887729749083519, -0.22571922838687897, -0.2964034378528595, -0.2490723431110382, -0.014802857302129269, -0.6396473050117493, 0.8369805216789246, 0.0007463748333975673, -0.07657469809055328, 0.5500116944313049, -0.0001780646271072328, 0.3117384910583496, 0.7270472049713135, -0.12233739346265793, -0.17463967204093933, 0.38583576679229736], [-0.07800176739692688, -0.05697914585471153, 0.10527446866035461, -0.016361350193619728, -0.15798676013946533, 0.013820475898683071, 0.2730811536312103, 0.0576886348426342, -0.09511792659759521, 0.1933211088180542, 0.5603917837142944, -0.1260940581560135, -0.015168950892984867, 0.34067150950431824, 0.010290976613759995, 0.017842747271060944, -0.0020353770814836025, -0.007928116247057915, -0.047707103192806244, 0.0609915517270565, -0.05854572355747223, -0.4770466685295105, 0.1341893970966339, -0.0007764574838802218, -0.07040992379188538, 0.027324222028255463, 0.010501338168978691, -0.1699804961681366, -0.44854986667633057, 0.029624000191688538, -0.015154220163822174, -0.005981624126434326], [0.011811280623078346, -0.4026007652282715, -0.05723925679922104, -0.016004152595996857, 0.10806723684072495, 0.013488358817994595, 0.04400563985109329, 0.18963149189949036, 0.025981497019529343, -0.3951292335987091, -0.6346657276153564, -0.6330483555793762, 0.03621737286448479, -0.041416507214307785, 0.010405225679278374, -0.033953599631786346, -0.1225733831524849, -0.17000722885131836, 0.08714233338832855, 0.39143067598342896, -0.1617041677236557, -0.11862853914499283, -0.15508131682872772, 0.0014683003537356853, 0.022715583443641663, 0.20561198890209198, 0.010855494998395443, 0.15981827676296234, 0.04518590494990349, -0.1537744551897049, 0.5880041122436523, 0.21104764938354492], [0.025415722280740738, 0.006192485801875591, 0.024344753473997116, 0.0010907528921961784, 0.019402818754315376, -0.0007461636560037732, 0.06701526045799255, -0.007767657749354839, -0.06670163571834564, 0.057894036173820496, -0.01076519675552845, -0.07919393479824066, -0.0009229190181940794, 0.006975743919610977, -0.0003358348039910197, 0.0287541002035141, -0.0019939509220421314, 0.007277173921465874, 0.057841505855321884, -0.00390078267082572, 0.06895242631435394, 0.07333401590585709, 0.08883234113454819, 0.0012068056967109442, -0.06650736927986145, -0.009109320119023323, -0.0003812365757767111, 0.018345115706324577, -0.05359765514731407, -0.009085594676434994, -0.005269197281450033, -0.18116632103919983], [-0.2437923103570938, -0.0033414822537451982, -0.056113213300704956, 4.3924490455538034e-05, -0.06300247460603714, -0.00032859283965080976, -0.07923772186040878, 0.02305195853114128, 0.21706050634384155, -0.2444029599428177, 0.05283104255795479, 0.2020513266324997, 0.002513331826776266, -0.05669942498207092, -0.0007508855778723955, -0.0634051039814949, 0.01436333917081356, -0.015326099470257759, -0.16433513164520264, 0.06359705328941345, -0.18201331794261932, -0.09757930785417557, -0.37490972876548767, -0.0024677354376763105, 0.2306794822216034, -0.0007708182092756033, -0.0007212321506813169, -0.06765597313642502, 0.15852884948253632, 0.03770984709262848, -0.001597795751877129, 0.46188679337501526], [-0.30341315269470215, 0.06814289093017578, -0.013445926830172539, -0.0006937147118151188, 0.06079074367880821, 0.0005704787326976657, 0.08298071473836899, -0.047773491591215134, 0.028148334473371506, 0.13153377175331116, 0.11869756132364273, -0.037608008831739426, 0.0045609246008098125, 0.23694384098052979, -9.26805951166898e-05, -0.12024638801813126, -0.01852985844016075, -0.00584187638014555, 0.26598304510116577, -0.3785548210144043, -0.05143284797668457, -0.1582244485616684, -0.5186628699302673, 0.005886891856789589, -0.2617204785346985, -0.3711843490600586, 0.0006044263718649745, 0.06929575651884079, -0.08859468996524811, -0.04517928138375282, -0.10784005373716354, 0.08585143834352493], [0.06282693892717361, 0.0423700325191021, -0.14161191880702972, -0.013512000441551208, 0.0018219087505713105, 0.011557736434042454, -0.041442494839429855, -0.008075806312263012, 0.04485471546649933, 0.05877752974629402, -0.09416395425796509, 0.11828978359699249, -0.01634843461215496, -0.007865596562623978, 0.008560696616768837, 0.016196822747588158, 0.00672560278326273, -0.0025160987861454487, 0.004860045854002237, -0.0885867029428482, -0.07750947028398514, 0.1333940178155899, -0.02753559499979019, -0.0024717096239328384, -0.08994143456220627, -0.023839954286813736, 0.008730035275220871, 0.021126648411154747, 0.11821072548627853, 0.022121261805295944, 0.038119226694107056, -0.026157014071941376], [0.22635632753372192, -0.11979203671216965, 0.009293540380895138, 0.022684384137392044, 0.025017468258738518, -0.020353922620415688, -0.20444966852664948, 0.2843572497367859, -0.009720963425934315, -0.28500452637672424, 0.2992648482322693, -0.26006564497947693, 0.048267848789691925, 0.36846908926963806, -0.017521828413009644, -0.09757594019174576, 0.043999914079904556, 0.1329992562532425, 0.031187422573566437, 0.13908065855503082, -0.04430956020951271, 0.28623881936073303, -0.12058784812688828, -0.00911488477140665, 5.154447353561409e-05, 0.08752982318401337, -0.017791101709008217, 0.23956269025802612, 0.3240932822227478, 0.03759016841650009, 0.03644484281539917, -0.01588345691561699], [-0.35917744040489197, -0.05155033618211746, -0.14974942803382874, 0.0017187395133078098, 0.165317103266716, -0.003430646378546953, 0.11806009709835052, -0.02315947785973549, 0.12553848326206207, 0.10957862436771393, 0.26737645268440247, 0.12445199489593506, -0.014919332228600979, -0.12184679508209229, -0.004637539386749268, 0.2752546966075897, -0.05792929232120514, -0.07565612345933914, -0.00614563561975956, 0.08444619923830032, 0.24908187985420227, 0.027100440114736557, 0.13571977615356445, -0.004552335944026709, 0.28667572140693665, 0.012695681303739548, -0.004397247452288866, 0.16829586029052734, -0.20679767429828644, -0.07528357207775116, -0.004295028280466795, 0.07909485697746277], [0.026041200384497643, 0.05858999863266945, 0.05200578644871712, -0.04479056969285011, 0.1822432577610016, 0.040285874158144, 0.0057640597224235535, -0.1688646525144577, -0.036947380751371384, 0.11761809140443802, 0.16922666132450104, 0.0383278913795948, -0.03333066776394844, -0.03646284341812134, 0.03512445464730263, -0.4152718782424927, -0.10382328927516937, -0.09824193269014359, -0.4224544167518616, -0.1006447821855545, -0.36442655324935913, -0.004017628263682127, -0.044220492243766785, 0.01936502754688263, 0.12386088073253632, -0.2689385414123535, 0.03579629585146904, -0.22705833613872528, -0.024292564019560814, -0.15205572545528412, -0.1332903504371643, 0.34123918414115906], [-0.33131831884384155, -0.040775805711746216, -0.01754354126751423, 0.03153013810515404, -0.06640015542507172, -0.028758954256772995, -0.5408162474632263, 0.11161989718675613, -0.4743291139602661, -0.47226014733314514, -0.10736418515443802, -0.3855520188808441, 0.021224306896328926, 0.146537184715271, -0.025488976389169693, 0.15052585303783417, 0.08483029156923294, 0.048093877732753754, -0.46586063504219055, -0.549415111541748, 0.09991715848445892, 0.13263070583343506, 1.1069550514221191, -0.012783277779817581, -0.9161930680274963, -0.07439056038856506, -0.025856824591755867, 0.009621182456612587, -0.18598783016204834, 0.09442910552024841, 0.2074003964662552, 0.03787386789917946], [-0.030656598508358, 0.07287072390317917, 0.024258123710751534, -0.002656717784702778, -0.02454557456076145, 0.003032806795090437, 0.3479522168636322, -0.02136649191379547, 0.011555243283510208, -0.3799295723438263, 0.11955340951681137, 0.16734039783477783, -0.02239183709025383, -0.19618463516235352, 0.003536777338013053, -0.013293011114001274, 0.022396184504032135, 0.033981289714574814, -0.1107763722538948, -0.041117168962955475, 0.048336517065763474, 0.0577947162091732, 0.0773652195930481, 0.0036843318957835436, 0.029847538098692894, 0.06610918790102005, 0.0035074357874691486, -0.10078765451908112, 0.24877400696277618, 0.034111492335796356, -0.0998867079615593, -0.13531386852264404], [-0.11198528110980988, 0.08565770089626312, -0.014480678364634514, -0.004010454751551151, -0.04291478544473648, 0.003916043788194656, 0.14737018942832947, -0.02680950239300728, 0.23431286215782166, -0.34590065479278564, 0.21289043128490448, 0.21671918034553528, -0.019435536116361618, -0.1971830129623413, 0.0038358671590685844, -0.040804099291563034, 0.013767779804766178, 0.009778797626495361, -0.20310337841510773, -0.04995419457554817, -0.03186232969164848, -0.10700155049562454, 0.06122396141290665, 0.002185235731303692, 0.08729884773492813, 0.0345979779958725, 0.0038431580178439617, -0.126257985830307, 0.2520734667778015, 0.03213849291205406, -0.09802651405334473, -0.022040560841560364], [-0.14133165776729584, 0.05404885113239288, -0.038823094218969345, -0.002920666942372918, -0.0436275377869606, 0.002643550978973508, 0.05668816715478897, -0.013405753299593925, 0.27307143807411194, -0.2595471143722534, 0.1821892410516739, 0.20336021482944489, -0.010183591395616531, -0.14619417488574982, 0.0023147810716181993, -0.05224549025297165, 0.007448318414390087, -0.007646930869668722, -0.1673678159713745, -0.023020727559924126, -0.09324625134468079, -0.15012826025485992, -0.06721986085176468, 0.0002855409402400255, 0.12301399558782578, 0.006259358488023281, 0.0023387200199067593, -0.10127343982458115, 0.21029873192310333, 0.025502467527985573, -0.05838261917233467, 0.08896937221288681], [-0.13254989683628082, 0.026764461770653725, -0.04666123166680336, -0.0018759072991088033, -0.04036242887377739, 0.001547310152091086, 0.026529939845204353, -0.0014720528852194548, 0.23759038746356964, -0.21446582674980164, 0.12660859525203705, 0.1793319433927536, -0.004219206981360912, -0.095077283680439, 0.0011324520455673337, -0.053797584027051926, 0.005261612590402365, -0.01355015579611063, -0.17068244516849518, -0.0024881488643586636, -0.11683401465415955, -0.1945594698190689, -0.1517696976661682, -0.0008637309074401855, 0.13268722593784332, 0.11034446209669113, 0.0011646043276414275, -0.07235704362392426, 0.16230039298534393, 0.02146408148109913, -0.027288265526294708, 0.17409618198871613], [-0.0135677894577384, 0.03657636418938637, -0.14692282676696777, -0.01154874637722969, -0.08601384609937668, 0.010739541612565517, 0.4043941795825958, 0.01007306482642889, 0.5097440481185913, 0.04126441106200218, 0.10694944858551025, 0.05504164099693298, -0.0393848717212677, 0.10467154532670975, 0.010721406899392605, 0.332073837518692, 0.03497670218348503, -0.08827096223831177, 0.9140632748603821, -0.1700243502855301, 0.2874685525894165, 0.15853165090084076, -0.3438253402709961, 0.007208232767879963, -0.24792461097240448, -0.1296289563179016, 0.010139756835997105, -0.031504809856414795, 0.025618799030780792, 0.07762756198644638, 0.07076077908277512, -0.12329129874706268], [0.25636953115463257, 0.15928757190704346, 0.016984134912490845, -0.010508023202419281, -0.17906251549720764, 0.009438108652830124, -0.1418178677558899, 0.050469763576984406, 0.019391153007745743, 0.10555263608694077, -0.04798645153641701, 0.11440813541412354, -0.006480904761701822, 0.1570218801498413, 0.008113246411085129, 0.30894872546195984, 0.11090294271707535, 0.16865399479866028, -0.08557421714067459, -0.09147018194198608, 0.43012475967407227, 0.2558331787586212, 0.3746587634086609, -3.622699296101928e-05, 0.2482922077178955, -0.25621411204338074, 0.008266459219157696, 0.003114799503237009, -0.02167641371488571, 0.18493160605430603, -0.19359810650348663, -0.166032612323761]],dtype=fnp.float32)
        neural_1=fnp.asarray([0.6666090488433838, 0.028226934373378754, 0.02508426271378994, -0.0025780347641557455, 0.17100027203559875, 0.0026825889945030212, 0.05897422507405281, -0.06771956384181976, -0.40006038546562195, 0.45718374848365784, -0.22406166791915894, -0.2900627553462982, -0.0077909245155751705, 0.22867608070373535, 0.0029419241473078728, 0.09979765862226486, -0.05092501640319824, 0.010246304795145988, 0.26278698444366455, -0.18364134430885315, 0.3147411048412323, 0.17404483258724213, 0.5941264033317566, 0.003641590243205428, -0.5202480554580688, -0.0032443038653582335, 0.002998526906594634, 0.2326323688030243, -0.32487767934799194, -0.10727398097515106, 0.013054639101028442, -0.5980011224746704],dtype=fnp.float32)
        neural_2=fnp.asarray([[0.0035437436308711767, -0.0050240783020854, 0.39631032943725586, 0.007449285592883825, 0.0017232773825526237, 0.008014240302145481, 0.0026084717828780413, 0.003241520142182708, -0.4001295268535614, -0.001524779130704701, -0.0025093615986406803, 0.0038561660330742598, -0.006422833539545536, -0.0010002965573221445, -0.0010405783541500568, 0.1693340241909027, -0.00011343374353600666, -0.007121937349438667, 0.00029990365146659315, -0.003002173500135541, 0.0018960810266435146, 0.0028759080450981855, 0.1389719396829605, -0.005642641335725784, -0.07029649615287781, 0.0015522404573857784, 0.08472602069377899, 0.0025246802251785994, -0.41560661792755127, 0.005322638433426619, -0.0033960756845772266, 0.007664908654987812], [0.003005852224305272, -0.004142051562666893, 0.41820862889289856, 0.0023107060696929693, 0.0037568877451121807, 0.003408141201362014, -0.01620839722454548, 0.00694661820307374, 0.003551302943378687, 0.005997646600008011, 0.011180118657648563, 0.00515490910038352, -0.0036564874462783337, -0.005612912587821484, -0.0023312997072935104, -0.1608406901359558, 0.002713749185204506, -0.0036558315623551607, 0.0027511813677847385, 0.008420057594776154, 0.008659451268613338, 0.005802774801850319, -0.1287890523672104, -0.0038196188397705555, 0.12169267237186432, -0.00580383138731122, -0.22552764415740967, 0.005653585772961378, -0.04802502319216728, 0.002909853123128414, -0.0033445756416767836, 0.002263458212837577], [-0.005704144947230816, 0.002324612345546484, -0.07897792011499405, -0.0027439252007752657, 0.0004092626040801406, -0.0025056705344468355, 0.0005713997525162995, -0.00496318656951189, 0.12558218836784363, -0.00081836455501616, -0.0010997923091053963, -0.002489939797669649, 0.00299090682528913, -0.006088707130402327, 0.002789541147649288, -0.24851155281066895, -0.003330199047923088, 0.003477594582363963, -0.0013297521509230137, -0.0023244500625878572, -0.006195181980729103, -0.004937442019581795, -0.01836545020341873, -0.006113606970757246, -0.03285188600420952, 0.0016154906479641795, -0.05482013523578644, -0.0019521277863532305, 0.038973368704319, -0.0024601835757493973, 0.005627835635095835, -0.0022741034626960754], [-0.001260176533833146, -0.0009475069819018245, -0.016260487958788872, -0.0025205472484230995, -0.0017493714112788439, -0.0023457775823771954, 0.002265379298478365, -0.005091242957860231, -0.001453581964597106, -0.00118820124771446, -0.003767921356484294, -0.0017672475660219789, 0.002478287322446704, 0.0008976642275229096, 0.0032405515667051077, -0.010562681593000889, -0.006093964446336031, 0.0027775741182267666, -0.001217271201312542, -0.007631832733750343, -0.005071201827377081, -0.004402667749673128, 0.022516200318932533, 0.0015676106559112668, -0.060575298964977264, 0.0024914781097322702, -0.03956286981701851, -0.0007640618714503944, 0.01628926210105419, -0.002294511068612337, 0.0013039126060903072, -0.0020799143239855766], [0.0038978513330221176, 0.0008335410384461284, -0.029574254527688026, 0.0030997712165117264, -0.0014921324327588081, -0.00038106448482722044, -0.003982176538556814, 0.002316477708518505, 0.02395392581820488, 0.0020723086781799793, 0.0006906743510626256, 0.0026247315108776093, -0.0020323642529547215, 0.004978694953024387, -0.00446291221305728, 0.2715904116630554, -0.002542204689234495, -0.0018049877835437655, 0.00034997810143977404, -0.003798271529376507, 0.002807312412187457, 0.0029369124677032232, 0.022659359499812126, 0.000726803089492023, 0.07707062363624573, -0.0013421080075204372, 0.012615863233804703, 0.0027223117649555206, -0.21762199699878693, 0.0007003085338510573, -0.003650491591542959, 0.001365784090012312], [0.0010483062360435724, 0.0001469191920477897, 0.01402160245925188, 0.0025133120361715555, 0.0018528441432863474, 0.0023851271253079176, -0.0025208909064531326, 0.004975747317075729, 0.0012769821332767606, 0.0012224990641698241, 0.003758891485631466, 0.0017513164784759283, -0.002465511905029416, -0.0009309485321864486, -0.003241673344746232, 0.009193167090415955, 0.005876652430742979, -0.002723014447838068, 0.0012420350685715675, 0.007407538127154112, 0.004990389570593834, 0.004343884065747261, -0.02091989479959011, -0.001563575933687389, 0.05494105815887451, -0.0024759341031312943, 0.03543230891227722, 0.0007569731678813696, -0.014964910224080086, 0.00234213937073946, -0.0010836598230525851, 0.002125738887116313], [-0.002383672632277012, -0.013595103286206722, -0.15895293653011322, -0.0058747995644807816, -0.0029601280111819506, -0.005395733285695314, -0.009499525651335716, -0.004350040107965469, -0.03515561297535896, 0.003564255079254508, 0.00499475933611393, -0.005152244586497545, 0.005939069669693708, 0.005321537610143423, 0.0018168833339586854, 0.2521475851535797, -0.004113968461751938, 0.006973720155656338, 0.0015260815853253007, -0.007043752353638411, -0.0036249400582164526, -0.002805214375257492, -0.32870548963546753, 0.011553269810974598, 0.15449278056621552, -0.0009889383800327778, -0.2797195017337799, -0.006784060504287481, 0.2690945267677307, -0.004411864094436169, 0.0023957551456987858, -0.005274824798107147], [-0.0004189403261989355, 0.0017012662719935179, -0.334403932094574, -0.0005817990167997777, -0.0022271445486694574, 0.0002651758841238916, 0.018677690997719765, -0.005168258212506771, -0.019591279327869415, -0.006683824583888054, -0.010534754022955894, -0.0008530034101568162, 0.0011546778259798884, -6.323267007246614e-05, 0.004490237683057785, -0.20416122674942017, -0.0013720012502744794, 0.001039066119119525, -0.003753405762836337, -0.0007123241084627807, -0.005805973894894123, -0.003642162075266242, 0.07258301228284836, 8.937520760810003e-06, -0.144252747297287, 0.005837282165884972, 0.08954647183418274, -0.0007819510065019131, 0.17242173850536346, -0.00030323173268698156, 0.0001021792704705149, -9.088862861972302e-05], [0.004311537370085716, -0.01422938983887434, 0.1001347005367279, 0.001954649342224002, 0.0011454866034910083, 0.001572505454532802, 0.003103946102783084, 0.0021160515025258064, -0.30557671189308167, 0.0010907943360507488, 0.0016368977958336473, 0.004578243009746075, -0.002114366739988327, -0.003174828365445137, -0.0008189535583369434, 0.30544328689575195, 0.0018089249497279525, -0.002389000728726387, -5.849722583661787e-05, 0.007269863039255142, 0.0021224170923233032, 0.0014757727039977908, -0.448813796043396, -0.00641776854172349, 0.32310885190963745, -0.0016296417452394962, -0.35420629382133484, 0.004189626779407263, 0.45499128103256226, 0.0011012402828782797, -0.004226098768413067, 0.001522734179161489], [0.0006062212632969022, 0.019521865993738174, 0.4216606616973877, 0.0022884788922965527, 0.0005896779475733638, 0.000500370457302779, -0.0017819026252254844, 0.00187198375351727, -0.06048576161265373, -0.00026635275571607053, 0.0015134693821892142, -0.0007941147196106613, -0.0023524693679064512, 0.003726235358044505, -0.002451487584039569, -0.09122135490179062, 0.005132193211466074, -0.0023443729151040316, 0.0023168076295405626, 0.0011079161195084453, 0.003357204142957926, 0.006030517630279064, 0.28885510563850403, 0.003081610659137368, -0.19949808716773987, -0.0009885553736239672, 0.3382495939731598, -0.0004517158668022603, -0.5466598868370056, 0.002229164820164442, -0.00033305829856544733, 0.0015737551730126143], [0.01024541538208723, -0.005182995460927486, 0.15693596005439758, 0.01049723755568266, 0.0014522451674565673, 0.005209670402109623, -0.004267867188900709, 0.010862994007766247, 0.4707336723804474, 0.002918076002970338, 0.004516137298196554, 0.005535319913178682, -0.0077669695019721985, 0.009633488021790981, -0.011641547083854675, -0.3607006371021271, 0.009971016086637974, -0.00827872846275568, 0.003935340791940689, 0.008249705657362938, 0.01064422819763422, 0.011086811311542988, -0.18409286439418793, 0.005369609221816063, 0.18224164843559265, -0.005570319481194019, -0.1932487040758133, 0.004094511736184359, 0.08767428249120712, 0.00567682133987546, -0.00901985913515091, 0.007126538548618555], [0.0013911074493080378, -0.005997688043862581, 0.5311363339424133, 0.0026872388552874327, 0.0006291103782132268, 0.00039121220470406115, 0.0011181074660271406, 0.0019751903600990772, 0.21332307159900665, -0.0019834793638437986, -0.000882957479916513, 0.000178015063283965, -0.0013442597119137645, 0.0022981842048466206, 0.00013327310443855822, -0.265133261680603, 0.0042791119776666164, -0.000851383781991899, 0.0006966370856389403, 0.003671702230349183, 8.923080167733133e-05, 0.001154440687969327, -0.10445381700992584, 0.002123900456354022, 0.10685903578996658, 0.0013224739814177155, -0.0916278213262558, -0.0014001191593706608, -0.03431113436818123, 0.0014081811532378197, -0.00039678547182120383, 0.0011962479911744595], [0.0006278497166931629, 0.0007770142983645201, -0.04635030776262283, -0.002267241245135665, -0.0020937398076057434, -0.0025108857080340385, 0.005811790004372597, -0.004737306851893663, -0.010853455401957035, -0.0012746516149491072, -0.00484320567920804, -0.0017319242469966412, 0.002312456024810672, 0.001366815180517733, 0.003160861786454916, -0.010020432993769646, -0.005711714271456003, 0.002285200171172619, -0.0013885051012039185, -0.007764957845211029, -0.005017987918108702, -0.00444052554666996, 0.012166616506874561, 0.0016181061509996653, -0.07675309479236603, 0.002432223642244935, -0.005635485984385014, -0.0009317573276348412, 0.012374628335237503, -0.002571409335359931, -0.0005163611494936049, -0.002175102708861232], [0.003766582580283284, -0.0014963980065658689, -0.06233939155936241, -0.0009585176012478769, 0.0020108181051909924, 4.6576285967603326e-05, -0.0014388620620593429, 7.393877604044974e-05, -0.5481524467468262, 0.0023518032394349575, 0.0007869868422858417, 0.003122080350294709, -0.001251758774742484, -0.0008634735131636262, 0.0007714349194429815, 0.023448383435606956, -0.004899479448795319, -0.0017423619283363223, 0.0009822206338867545, -0.003362925723195076, 0.0015036684926599264, -0.0012326156720519066, 0.08980661630630493, -0.0004899452906101942, -0.10772031545639038, -0.0012384274741634727, 0.12546101212501526, 0.0032457360066473484, -0.18459193408489227, -0.0002040812250925228, -0.0030371856410056353, -0.0003006374463438988], [0.0009896708652377129, -0.0009559483733028173, 0.011428281664848328, 0.0026831466238945723, 0.0019488102989271283, 0.0025226492434740067, -0.003076319582760334, 0.005148403812199831, 0.001005004858598113, 0.0013330959482118487, 0.0038954257033765316, 0.0018310031155124307, -0.0025884276255965233, -0.0007909539272077382, -0.00349987018853426, 0.007629487197846174, 0.005867751315236092, -0.002809450961649418, 0.0013059602351859212, 0.007348118349909782, 0.005199842620640993, 0.004544933792203665, -0.01926652528345585, -0.001409013639204204, 0.048836033791303635, -0.0025594986509531736, 0.03101571463048458, 0.0007592595648020506, -0.013687909580767155, 0.0025179486256092787, -0.0010131499730050564, 0.0022971872240304947], [-0.006139294244349003, -0.004396575968712568, 0.12650759518146515, -0.004213202279061079, 0.00595458596944809, -0.0002845373237505555, -0.00269276462495327, -0.004411435220390558, -0.3569735288619995, 0.003469301387667656, 0.005174562335014343, -0.004508799873292446, 0.0033283194061368704, -0.008779311552643776, 0.005626196041703224, 0.07982668280601501, 0.0008462389814667404, 0.0034378678537905216, 0.003706896211951971, 0.0016809962689876556, -0.003381195245310664, -0.0034988028928637505, 0.08310104161500931, -0.003280067816376686, -0.11159781366586685, -0.0035960664972662926, 0.17810241878032684, -0.003219262231141329, 0.41313546895980835, -0.0017000740626826882, 0.006257480010390282, -0.0026600216515362263], [-0.00025537103647366166, -0.00368528557009995, 0.005572841037064791, 0.0010338604915887117, 0.002130660228431225, 0.0017233745893463492, 0.0020440337248146534, 0.00019118544878438115, -0.016307493671774864, -0.000261505541857332, -8.368547423742712e-05, 0.0014515313087031245, -0.0014199811266735196, -0.003637677989900112, 0.0005655624554492533, -0.17906159162521362, 0.00044615170918405056, -0.001399757107719779, 9.95559457805939e-05, 0.002589002950116992, 0.00015565817011520267, 0.00048243801575154066, -0.013595778495073318, -0.003740417305380106, -0.08701558411121368, -0.000300521613098681, -0.05639222636818886, 0.0020366404205560684, 0.10578431189060211, 0.0013542742235586047, 0.00022586906561627984, 0.0013432875275611877], [0.0011573167284950614, 0.0009923080215230584, -0.0638953223824501, -0.0006797389942221344, 0.0006349201430566609, -0.0006394182564690709, 0.008229798637330532, -0.0030371069442480803, -0.011408069171011448, -0.00044230715138837695, -0.0034884773194789886, 0.00027759186923503876, 0.0006909740623086691, -0.0006809092010371387, 0.0015673182206228375, -0.31191208958625793, -0.003570114728063345, 0.0005090224440209568, -9.806851448956877e-05, -0.001162885338999331, -0.0024944583419710398, -0.001548871980048716, -0.013473039492964745, -0.0003580771153792739, -0.06773284822702408, 0.0004527313285507262, -0.06184833496809006, 0.0009119064779952168, 0.11416732519865036, -0.0005595650873146951, -0.0011568047339096665, -0.0006576101295650005], [-0.00718487985432148, 0.01718825101852417, 0.07627058029174805, -0.004472386557608843, 0.0006299954256974161, -0.002666810527443886, 0.005264454986900091, -0.006805869750678539, 0.9777851700782776, -0.000955818744841963, -0.0032463304232805967, -0.009843122214078903, 0.005031786859035492, 0.010904189199209213, 9.100958413910121e-05, -0.442219614982605, 0.0012350872857496142, 0.006132146343588829, 0.0042923386208713055, -0.007730576675385237, -0.005614456254988909, -0.0032105755526572466, 0.07414095103740692, 0.013203790411353111, -0.1530173271894455, -0.000671232002787292, 0.16096697747707367, -0.009545117616653442, -0.5718151926994324, -0.0019383372273296118, 0.008368037641048431, -0.0029643485322594643], [-0.003655072068795562, -0.007928619161248207, -0.5257512331008911, -0.0027311325538903475, 0.0025532389990985394, 0.00040608839481137693, 0.0021310290321707726, -0.002336474834010005, 0.4149068295955658, -0.0011129797203466296, 0.00018285814439877868, -0.0009077670983970165, 0.0011137975379824638, -0.003645269200205803, 0.00026724746567197144, -0.12361995875835419, -0.0006505013443529606, 0.0004159036325290799, 0.00039231046685017645, 0.0006203835364431143, 0.00032244669273495674, -0.0008491455810144544, -0.1174042597413063, -0.004240154754370451, 0.05655759945511818, 0.0006511135725304484, -0.011628959327936172, -0.0012208354892209172, 0.1387605518102646, -0.00033942036679945886, 0.00474396301433444, -0.0010481096105650067], [-0.001446866663172841, -0.003989273216575384, 0.2066650539636612, -0.00044689589412882924, 0.0039884052239358425, 9.126687655225396e-05, -0.00865691527724266, -0.0007331061642616987, -0.5835948586463928, 0.005979851819574833, 0.007107321172952652, -0.00036679324693977833, 0.0007142657996155322, -0.006694431882351637, 0.001493140240199864, 0.13827373087406158, -0.00173089187592268, 0.000748133403249085, 0.003685503965243697, -0.0006673848838545382, -3.1637115171179175e-05, -0.0007277780096046627, 0.02922041527926922, -0.005424042232334614, -0.043217774480581284, -0.00553167937323451, 0.07043859362602234, 0.0010390984825789928, 0.2146115005016327, -0.0005372888408601284, 0.0014892200706526637, -0.0007637693779543042], [0.0032715958077460527, 0.002660829806700349, -0.20052912831306458, 0.006449083331972361, 0.0010624127462506294, 0.004840591456741095, 0.010963212698698044, 0.0012625459348782897, 0.8358801007270813, -0.003089139237999916, -0.007467726245522499, 0.004624463617801666, -0.005848526954650879, 0.005582031793892384, -0.004257743712514639, -0.3034727871417999, -0.0011191593948751688, -0.006092710420489311, -0.0004015423764940351, -0.0023049693554639816, 0.001948182238265872, 0.0027545979246497154, 0.21723152697086334, -0.0034196223132312298, -0.17451795935630798, 0.0020208796486258507, 0.16825169324874878, 0.005646764300763607, -0.16671410202980042, 0.0045758020132780075, -0.0027062157168984413, 0.005963756237179041], [-0.006109069101512432, -0.04065990820527077, -0.1604764312505722, -0.004991575144231319, 0.0008305764058604836, -0.002939528552815318, 0.0019907024689018726, -0.006578733213245869, -0.3652435541152954, 0.000694066344294697, -0.002290359465405345, -0.008439877070486546, 0.006250799633562565, 0.007517603226006031, -0.0003319901879876852, 0.4570157527923584, -0.0019656403455883265, 0.007360023446381092, 0.0033124363981187344, -0.007184068206697702, -0.005263775587081909, -0.0034020787570625544, -0.35213935375213623, 0.011928193271160126, 0.3270868957042694, -0.000429769279435277, -0.42203521728515625, -0.009898321703076363, 0.3882736265659332, -0.0023208155762404203, 0.007299121469259262, -0.003460084553807974], [-0.0002901908301282674, -0.0055784075520932674, 0.00098267977591604, 0.0019518823828548193, 0.0024733783211559057, 0.0021074791438877583, -0.005090135149657726, 0.00419046962633729, -0.001031342544592917, 0.0020139431580901146, 0.004546962212771177, 0.00073428318137303, -0.0017641882877796888, 3.144394577248022e-05, -0.0034060569014400244, 0.0018641902133822441, 0.004899628926068544, -0.0017177659319713712, 0.0020845914259552956, 0.00563780078664422, 0.004545537754893303, 0.004010315053164959, -0.011125007644295692, 0.0002401342790108174, 0.027632934972643852, -0.0029069671873003244, 0.014644850045442581, -0.0004821483453270048, -0.008387311361730099, 0.0022209908347576857, 0.00047149480087682605, 0.0018125784117728472], [-0.0012248004786670208, -0.005303713493049145, 0.0901702418923378, -0.004919732920825481, 0.0003125369257759303, -0.0036926325410604477, 0.0012652907753363252, -0.0033543379977345467, 0.5359607338905334, -0.00012550475366879255, 0.0013677291572093964, 0.00014319911133497953, 0.0024387973826378584, -0.004385230131447315, 0.00428748456761241, -0.33069053292274475, -0.0022568374406546354, 0.0031414341647177935, -0.00040578353218734264, 0.003893730230629444, -0.0037824027240276337, -0.0027845995500683784, -0.10634779185056686, -0.0040574208833277225, 0.05972539633512497, 0.0010670716874301434, -0.07752113789319992, -0.0008247583755291998, 0.104129359126091, -0.003814246039837599, 0.001572753768414259, -0.003694192273542285], [-0.006870681885629892, 0.03930213674902916, 0.15678010880947113, -0.0054286690428853035, 0.0012455967953428626, -0.0011957983952015638, 0.0048444150015711784, -0.005642502103000879, 1.0698755979537964, 0.0001578623487148434, -0.0018476496916264296, -0.0056777591817080975, 0.0045984648168087006, -0.0007281505386345088, 0.002733272034674883, -0.46850210428237915, -0.0018584190402179956, 0.00557774817571044, 0.0008065414149314165, -0.00302520371042192, -0.004670448135584593, -0.004131990950554609, 0.2795140743255615, 0.003314381930977106, -0.2724495828151703, -0.000522201182320714, 0.39638951420783997, -0.004790652077645063, -0.3347490131855011, -0.0015365128638222814, 0.005985600873827934, -0.002750837244093418], [0.0009513197583146393, -0.000868318893481046, 0.011638682335615158, 0.0026264486368745565, 0.0019529322162270546, 0.0024879719130694866, -0.0030106829944998026, 0.005062488839030266, 0.0010131471790373325, 0.00131979922298342, 0.0038621884305030107, 0.0017899575177580118, -0.002540935529395938, -0.0008250420214608312, -0.003422277746722102, 0.007838452234864235, 0.005804330110549927, -0.0027603108901530504, 0.00131076795514673, 0.007267259526997805, 0.005115736275911331, 0.004470997955650091, -0.019127201288938522, -0.0014260923489928246, 0.04952199012041092, -0.002538324799388647, 0.03134618699550629, 0.0007429259130731225, -0.013839900493621826, 0.002474065637215972, -0.0009738260414451361, 0.0022519638296216726], [0.007160982117056847, -0.004182456526905298, -0.1522875726222992, 0.0036733930464833975, -0.0036789667792618275, -0.0007421359186992049, 0.007676532492041588, 0.0011041255202144384, -0.370917946100235, -0.0008786935359239578, -0.0039012294728308916, 0.002450499916449189, -0.0017766337841749191, 0.0064949230290949345, -0.000528200704138726, 0.16426032781600952, -0.00036496069515123963, -0.0019552537705749273, -0.0009683857206255198, -0.0011136289685964584, 0.0005685125943273306, 0.001372392987832427, 0.054960474371910095, 0.00458993436768651, -0.027277618646621704, 0.0007084849057719111, 0.06693442910909653, 0.0035057018976658583, -0.19668048620224, 0.00016492218128405511, -0.0075409188866615295, 0.000588578637689352], [0.0017294351710006595, -0.012251554988324642, 0.3049067556858063, -2.2283005819190294e-05, -0.0009399380069226027, -0.0014784481609240174, 0.008381757885217667, -0.0004076695186085999, -0.6104892492294312, -0.002626233035698533, -0.0076153879053890705, -0.0013197558000683784, 0.0015099557349458337, 0.0008215288398787379, -0.0015924671897664666, 0.4799911081790924, -0.003077921224758029, 0.0007513209711760283, -0.00027471184148453176, -0.005916391033679247, -0.0017486329888924956, -0.002106515923514962, -0.3678561747074127, -0.001793849398382008, 0.05402873456478119, 0.0024803795386105776, -0.35472655296325684, -0.0014348849654197693, 0.23282502591609955, -0.001638326677493751, -0.0006751709152013063, -0.000645176914986223], [-0.0010138435754925013, -0.0025019010063260794, 0.01046382263302803, -0.0004997726646251976, 0.0017596223624423146, 0.0011666037607938051, 0.0030762350652366877, -0.0008453845512121916, -0.035060446709394455, -0.0003914578119292855, -4.6084722271189094e-05, -0.00034165894612669945, 9.283499093726277e-05, -0.0027787021826952696, 0.001736281206831336, -0.26332390308380127, 0.0011335923336446285, -5.8577541494742036e-05, 0.00037732510827481747, 0.0027536971028894186, -0.0004964782274328172, -0.00032687163911759853, -0.030728287994861603, -0.0010488342959433794, -0.07557128369808197, -0.00028523243963718414, -0.0359143391251564, 0.00015356874791905284, 0.1692420393228531, 0.0006517705623991787, 0.001028649159707129, 0.0002305427478859201], [-0.004094449337571859, -0.0022251205518841743, -0.39290744066238403, -0.002246666233986616, 3.1365518225356936e-05, 0.00019303985754959285, 0.009657764807343483, -0.003592581255361438, -0.13763001561164856, -0.004359761252999306, -0.006701684091240168, -0.0020609195344150066, 0.0013026518281549215, -0.008069606497883797, 0.005356488283723593, 0.24921055138111115, -0.0027290862053632736, 0.0015597341116517782, -0.0019471434643492103, -0.003060863120481372, -0.00613374263048172, -0.005090683698654175, 0.17071247100830078, -0.006976534612476826, -0.15526165068149567, 0.004086918197572231, 0.26847052574157715, -0.0010312626836821437, 0.18205519020557404, -0.0005881711840629578, 0.0037807950284332037, -0.0003558245662134141], [-0.00021909743372816592, 0.02615867741405964, 0.013975042849779129, -0.002445741556584835, 0.00205808412283659, -0.0009413916268385947, -0.0019473331049084663, -0.0037039685994386673, 1.3171110153198242, 0.0015856371028348804, 0.002171936444938183, -0.00026455119950696826, 0.0014543175930157304, -0.007589881308376789, 0.006009581033140421, -0.5460131764411926, -0.004231705330312252, 0.0011789449490606785, 6.23190135229379e-05, -0.002128983847796917, -0.00266062468290329, -0.0031228912994265556, 0.3096486032009125, -0.006072538439184427, -0.28595149517059326, -0.0001261694123968482, 0.3120000660419464, 0.0003354204527568072, -0.917400598526001, -0.002607271308079362, 0.0004314710386097431, -0.002363000763580203]],dtype=fnp.float32)
        neural_3=fnp.asarray([0.00010125648259418085, 0.0070783463306725025, -0.0657186210155487, -0.0024739601649343967, -0.00208266731351614, -0.004102959763258696, -0.0009224567329511046, -0.00021736513008363545, -0.1490364670753479, -0.002481024479493499, -0.0004164589336141944, 0.0009988500969484448, 0.0030080811120569706, -0.002590309362858534, 0.0018092349637299776, 0.05178103223443031, -0.0026631299406290054, 0.003267431166023016, -0.002644071588292718, 0.0012501251185312867, -0.002329736016690731, -0.003253481350839138, 0.15029184520244598, -0.0035682059824466705, -0.08014687895774841, 0.0033835258800536394, 0.16005241870880127, -9.361703268950805e-05, -0.43425142765045166, -0.004325585439801216, 0.0005337808397598565, -0.0034178560599684715],dtype=fnp.float32)
        neural_4=fnp.asarray([-1.2270368655024715, -5.952584260529422, -0.4717390959258438, -3.6108459367772427, -5.092876298596431, -2.8310348742512383, 0.48745525795242445, 2.2671934332880914, -1.2083393096288042, -3.362443658358459, 0.3138286766936558, 1.2762776028005398, 0.5302791353864654, 0.7745924413236499, -1.8450663377287158, 0.4475431071295634, 2.789900551731154, 1.3260246923516608, -3.403768163842329, 3.3676371027806384, 1.7645621330279286, 3.419508135946717, -0.9357993295695805, 1.300607407790674, 0.33027972137585065, 2.992689662436524, 0.3465539251564599, 0.799036045854284, 1.1836425004118671, 4.305390607861314, 0.28694962023996334, -2.1960391165879045],dtype=fnp.float32)
        neural_5=fnp.asarray(0.0,dtype=fnp.float32)
        # Separate head/modulation constants for the last two layers (keyed by layers from the end).
        head_override={}
        head_override[0]=(0.26873454697910787,fnp.asarray([0.5764923818945318, -1.4695392340698885, -0.4061847171467427, 0.4375488636479895, 1.6081698269379088, -1.3560804108998197, -0.03647031963093032, -0.053806691928709974, 0.19658782506523148, 0.02829802357331784, -0.380756037905296, 0.1023772696366464, -0.4415519852752502, -0.852978578074347, -0.1121303719810989, 0.16621989805662404, 0.035744291303387896, -0.10635645174585424, 0.06846257362835324, -0.11007358336482503],dtype=fnp.float32),fnp.asarray([[-0.7991808773652548, 0.8475765138388343, -0.3468993557866989, 0.17158668448518916, 1.1205755499366141, 0.7391461573850934, -0.277290195771544, 0.0006001699114316569, 0.3701822277770159, -0.050972953409592583, -0.7880904674595579, 0.8086904614031629, 0.09698182154693752, -0.9891983094417107, 0.2306395977964228, -0.0012236054576034543, -0.24929990774826993, -0.2955666264746343, 0.42139117444445945, -0.2573812182198003], [0.8475765138388343, -0.20312832078425874, -0.09729075034678841, -0.028032868531491506, 0.008333451167524174, -0.04467418079264956, -0.0070891669843046665, 0.1732675030974843, -0.006333760909620473, -0.007633155101613111, -0.01616179280356823, -0.022731486953269493, -0.03591106302691922, -0.19347913830692343, -0.5025098390030701, -0.883579145353899, -0.920227115109527, -0.6835294229747707, -0.09276755238803985, -0.010062991512125452], [-0.3468993557866989, -0.09729075034678841, -0.3291793167227436, -0.099394589156589, -0.1454024349286711, -0.34908042510790394, 0.05132338818430456, 0.034845949579496556, -0.08877696339871838, 0.20126609372514886, 0.22667238595070574, -0.03547954304459167, -0.1887653332410613, -0.07041298736336919, 0.46440607598258765, 0.2149964440425067, 0.0678004753436508, -0.038727265186649526, 0.11407266384028643, 0.12795235930481696], [0.17158668448518916, -0.028032868531491506, -0.099394589156589, 0.0558966225938167, -0.13733295245462185, 0.2517913993417108, -0.061927612839752864, -0.03591060415407968, -0.009713735965225512, -0.03881337354084581, 0.10546235631934711, -0.08350829523072113, 0.08446504555177593, 0.09498105005003991, 0.5580787061522756, 0.049421257629078716, -0.056558995304713804, 0.03332967904740633, -0.2375331282709975, -0.13388847128643785], [1.1205755499366141, 0.008333451167524174, -0.1454024349286711, -0.13733295245462185, -0.034710384166981564, -0.007051989930601368, 0.1905468443252365, -0.234366558861216, 0.011864398009916803, -0.0010082193376053537, 0.005193644767615104, -0.030506837613832077, -0.01938732432709393, 0.01019075234221603, -1.8103689992700385, 0.37715769961704554, 2.2098467438646643, 2.6982360208384377, 0.01638344931302359, 0.0037071062113507936], [0.7391461573850934, -0.04467418079264956, -0.34908042510790394, 0.2517913993417108, -0.007051989930601368, 0.05235917855711255, -0.024378073564827626, 0.16425702513829268, -0.04569975902085573, -0.005459894806621126, 0.006064244961442054, 0.0034285079363267704, -0.01232258500214855, 0.014072328180897705, 0.5598632095202357, -1.3563756144976522, -1.2847528893358942, -0.7595213295949093, -0.04490595656480785, 0.07142103683683138], [-0.277290195771544, -0.0070891669843046665, 0.05132338818430456, -0.061927612839752864, 0.1905468443252365, -0.024378073564827626, -0.8337795218358257, 0.10632745509195865, -0.209847503479215, -0.03190580978046268, 0.2683749239745645, -0.2771106771036254, -0.023296238188444736, 0.029764018705774584, 0.010816936335978758, 0.04311704512752829, 0.07663486861355577, 0.13987827408789377, -0.006298274053934294, -0.25684777408736426], [0.0006001699114316569, 0.1732675030974843, 0.034845949579496556, -0.03591060415407968, -0.234366558861216, 0.16425702513829268, 0.10632745509195865, -0.03950485071941977, 0.02843195894738388, 0.0045086682430629564, -0.021264674573995064, 0.055897253654370156, 0.057243892848446026, 0.09227646995228939, 0.010440012242405317, -0.02995869962994709, -0.02293537580730223, -0.021833073612241793, -0.006449207126519858, 0.07573443150313612], [0.3701822277770159, -0.006333760909620473, -0.08877696339871838, -0.009713735965225512, 0.011864398009916803, -0.04569975902085573, -0.209847503479215, 0.02843195894738388, -0.017832443508576067, 0.0026187997021264483, 0.04934807290272096, 0.028409345362345698, 0.012492560779028581, 0.04816398609186601, -0.11844654620843287, -0.5274209217325951, -0.14397856226424305, 0.08246606704072051, -0.0482733996171654, -0.05507790644443762], [-0.050972953409592583, -0.007633155101613111, 0.20126609372514886, -0.03881337354084581, -0.0010082193376053537, -0.005459894806621126, -0.03190580978046268, 0.0045086682430629564, 0.0026187997021264483, 0.003826090392311821, -0.003990048618538191, -0.0168146063674713, -0.009908655762113723, 0.005592331992538439, -0.7737419847271483, 0.8375740075046043, 0.3066593933626194, -0.3093325115499461, -0.0012882554289848189, 0.07725352796640711], [-0.7880904674595579, -0.01616179280356823, 0.22667238595070574, 0.10546235631934711, 0.005193644767615104, 0.006064244961442054, 0.2683749239745645, -0.021264674573995064, 0.04934807290272096, -0.003990048618538191, -0.10632581285501051, 0.09648773211823355, -0.005421331899009696, -0.024669912756987454, 0.5340595191249723, -0.06127474374782209, -0.6065395700912222, -0.4003928273216553, 0.011879249849221535, 0.09695942006104935], [0.8086904614031629, -0.022731486953269493, -0.03547954304459167, -0.08350829523072113, -0.030506837613832077, 0.0034285079363267704, -0.2771106771036254, 0.055897253654370156, 0.028409345362345698, -0.0168146063674713, 0.09648773211823355, -0.02667873860319006, 0.03219159185069209, 0.005437286540121471, -1.1190278304578942, 0.7088337817232359, 0.11043120461560388, -0.4897676331513466, -0.018738272263101853, -0.18178199107096427], [0.09698182154693752, -0.03591106302691922, -0.1887653332410613, 0.08446504555177593, -0.01938732432709393, -0.01232258500214855, -0.023296238188444736, 0.057243892848446026, 0.012492560779028581, -0.009908655762113723, -0.005421331899009696, 0.03219159185069209, -0.032952737020694844, 0.008700859765580129, 0.6403101560873142, -0.6310831816451065, -0.5786937659025184, -0.19899083484063534, 0.12033502872724076, 0.1667980346761978], [-0.9891983094417107, -0.19347913830692343, -0.07041298736336919, 0.09498105005003991, 0.01019075234221603, 0.014072328180897705, 0.029764018705774584, 0.09227646995228939, 0.04816398609186601, 0.005592331992538439, -0.024669912756987454, 0.005437286540121471, 0.008700859765580129, -0.029398330562861463, 1.7106353741064448, -0.3035516049896053, -0.2243284015779092, 0.2134435288256211, 0.0751793184090196, 0.36605857497390865], [0.2306395977964228, -0.5025098390030701, 0.46440607598258765, 0.5580787061522756, -1.8103689992700385, 0.5598632095202357, 0.010816936335978758, 0.010440012242405317, -0.11844654620843287, -0.7737419847271483, 0.5340595191249723, -1.1190278304578942, 0.6403101560873142, 1.7106353741064448, 1.1263161672914384, 0.1181605447338849, -0.2504285142687528, -0.3673343721263161, -0.7063821181110248, 0.2053366142769433], [-0.0012236054576034543, -0.883579145353899, 0.2149964440425067, 0.049421257629078716, 0.37715769961704554, -1.3563756144976522, 0.04311704512752829, -0.02995869962994709, -0.5274209217325951, 0.8375740075046043, -0.06127474374782209, 0.7088337817232359, -0.6310831816451065, -0.3035516049896053, 0.1181605447338849, -0.7549985737973236, -0.6044347207888946, -0.5430338925604535, -0.6265239371718755, -0.271959956097361], [-0.24929990774826993, -0.920227115109527, 0.0678004753436508, -0.056558995304713804, 2.2098467438646643, -1.2847528893358942, 0.07663486861355577, -0.02293537580730223, -0.14397856226424305, 0.3066593933626194, -0.6065395700912222, 0.11043120461560388, -0.5786937659025184, -0.2243284015779092, -0.2504285142687528, -0.6044347207888946, -1.2655550112673686, -0.45732668837397655, 0.22022471941356586, -0.5133316251100954], [-0.2955666264746343, -0.6835294229747707, -0.038727265186649526, 0.03332967904740633, 2.6982360208384377, -0.7595213295949093, 0.13987827408789377, -0.021833073612241793, 0.08246606704072051, -0.3093325115499461, -0.4003928273216553, -0.4897676331513466, -0.19899083484063534, 0.2134435288256211, -0.3673343721263161, -0.5430338925604535, -0.45732668837397655, -0.5334708856652429, 0.5211026931410269, -0.43072460475104196], [0.42139117444445945, -0.09276755238803985, 0.11407266384028643, -0.2375331282709975, 0.01638344931302359, -0.04490595656480785, -0.006298274053934294, -0.006449207126519858, -0.0482733996171654, -0.0012882554289848189, 0.011879249849221535, -0.018738272263101853, 0.12033502872724076, 0.0751793184090196, -0.7063821181110248, -0.6265239371718755, 0.22022471941356586, 0.5211026931410269, -0.20067557969848465, -0.26776895864589983], [-0.2573812182198003, -0.010062991512125452, 0.12795235930481696, -0.13388847128643785, 0.0037071062113507936, 0.07142103683683138, -0.25684777408736426, 0.07573443150313612, -0.05507790644443762, 0.07725352796640711, 0.09695942006104935, -0.18178199107096427, 0.1667980346761978, 0.36605857497390865, 0.2053366142769433, -0.271959956097361, -0.5133316251100954, -0.43072460475104196, -0.26776895864589983, -0.21410444748100213]],dtype=fnp.float32),fnp.asarray([4.697340699880316, -3.5490371248594004, -0.291973105135473, -6.425588970811048, -9.431860517537778, -9.095427860751407, -0.4881657065573736, -0.34543389754808107, -1.2195766252992002, -0.5755739857143167, -0.12421609243342396, 2.7686366092916077, 0.3956567267550792, 3.917714761519832, 0.7901878992470128, 0.4630994555827472, -0.9042973983288878, 1.397985310829141, -2.396020275848251, 1.5497725782955265, 0.8682659965896714, 2.8873697294606533, -0.7904364182180161, 5.6802352072831575, 0.236929397145567, 6.87998989370036, 0.392665316015292, 5.32944280822104, 1.0544619963228792, 2.402482940257837, -5.907848312536041, -5.63403768008188],dtype=fnp.float32),[-0.058649044835996575, 0.31956308198512207, -0.16754327580692677, -0.13033768385417085, 0.007717849989942398, 0.006226924098386922, -0.06103487814197793, 0.041809036695249074, 0.0034551656268767585, 0.09185324089315773, -0.025061468953013564, 0.021849015665825637, -0.0242179839923168, 0.15696419253874452, -0.12446785477342873, -0.22746258207647255, -0.28427567264535847, 0.21917794777569582, 0.04732394750136232, -0.037177563119149874, -0.03207333563481896],[-0.3951889216791024, 0.15400512238969852, -0.14973877363335486, -0.07006663307846991, 0.010584714340368225, 0.04593350703839289, -0.06932369986505452, 0.08501389511099297, 0.07144213413319288, 0.06082245127371423, -0.03851898301071305, -0.04654595139833216, 0.038799799910371954, 0.09307294888154773, -0.07200262448712165, -0.3757421701791651, 0.5689954719963249, -0.3331024810942876, -0.7349016745297935, 0.10048601727085536, 0.3595637738628321],[4.893218073950463, -1.0628337181242258, 0.21512315666534376, -2.0293635320829546, 0.2140500961804328, -3.4233729702770557, 4.217775585106585, -0.39390229654985737, 0.13608890926346517, 0.09890508849047963, -0.6748446602907706, -2.3154782690900872, 1.4477578230053616, -0.6084210284567904, -3.866078512132892, 0.20639667241771736, 0.11970447235046844, 0.7563774922958633, 9.366165965276435, 0.4828457360997924, 1.8046542088497128, 2.61887699424772, 0.4160002959181084, 0.591512014076597, -0.22899129888676015, -0.284520474290646, 0.08125537314763853, -6.59150637621563, 0.4746954217564347, -0.8775181486493812, -2.2439586487649144, -1.0201496048146603],[-1.8892689330591332, 0.0307240299109037, 0.2111076608487974, 3.079857683124977, -0.5358357273821515, 4.2715217645494175, -2.6802814433558697, -1.3169784812887597, 0.125030996744035, -3.042712165027245, 0.26539511669392696, 0.5195245831760525, -2.8746475203427404, 5.508905321191987, 3.765208675834046, 0.208027729502903, 1.0705830614415346, -2.555510086019181, -3.66198098891347, -3.8838620799630377, -1.890669740098984, -0.2403599976754074, 0.16416924404061295, 3.074336704885353, -0.27370707313228293, 2.360152311597042, 0.007927344072316983, 0.5726220321681375, 0.45866172866098487, 2.289224446993164, 1.85563218780174, 2.7215273937162445])
        head_override[1]=(0.26893650061723023,fnp.asarray([0.6131566246580187, -1.445692500692678, -0.418026022543132, 0.4205223215057196, 1.6025967175020444, -1.3472719248341964, -0.033354484213619515, -0.054616712924262394, 0.21605404090829128, 0.022247021971638952, -0.4210405049210385, 0.15151893601816044, -0.44930129498802934, -0.8552230689864841, -0.11442658733015788, 0.16776195620373932, 0.03843488365859743, -0.10517990875188675, 0.06818106502813862, -0.02811775973178344],dtype=fnp.float32),fnp.asarray([[-1.1132003493928018, 0.831084629313427, -0.36577096586162033, 0.15111263132870645, 1.10981407521058, 0.7677013102662849, 0.005546368694395509, -0.0729286030117644, 0.37397354615613815, -0.08042620082195048, -0.7890830019183414, 0.8575548591940783, 0.10714702369206586, -0.9985761060440952, 0.2071414045284976, -0.03512783256485824, -0.3074825987636097, -0.3916090733553413, 0.43341345647119317, -0.1938256360252038], [0.831084629313427, -0.22469533360298344, -0.05238554871503123, -0.12989764653849256, 0.04215601161453669, -0.043100364265406484, 0.17687012505351465, 0.12544376772804366, 0.0181253535871363, 0.010153344712730047, -0.015401960775276346, -0.0945495534121372, -0.03337566055273211, -0.19810140354734807, -0.43489258065197106, -0.8463129620807677, -0.9168396051668359, -0.7116662481206637, -0.08455861051172718, -0.007065635528035637], [-0.36577096586162033, -0.05238554871503123, -0.2987943050626962, -0.09894527704194238, -0.13064382341179442, -0.36849626514810346, -0.040023217795586714, 0.05859324640108397, -0.09114539113414645, 0.23031401027628223, 0.23272173782282862, -0.16127433433163785, -0.1929221177333412, -0.07387062224176945, 0.4803604601132522, 0.23591221269865742, 0.09527325521337521, -0.006357075362910785, 0.11026004010550705, 0.14904182087895668], [0.15111263132870645, -0.12989764653849256, -0.09894527704194238, 0.04767886481221987, -0.12845541453917794, 0.30754213248167933, -0.19327435073721355, -0.0017645082161845335, -0.0069501138033697014, -0.047714669042042045, 0.07204534217944254, -0.06848418312183205, 0.10564390936983957, 0.05498202617474903, 0.5938346065585351, 0.08052089937825804, -0.025130725893016986, 0.07199755053201044, -0.24382388077351724, -0.13351492748106988], [1.10981407521058, 0.04215601161453669, -0.13064382341179442, -0.12845541453917794, -0.040937590279355784, 0.004959008094369536, 0.147554571568895, -0.22318989551544333, -0.004866175346369206, 0.0008987428749639296, 0.014561223366043909, -0.005112150405225184, -0.015786562371025447, 0.019490267635763996, -1.8007431311750377, 0.3704914794264029, 2.184304134923695, 2.6608310954027474, 0.014948852708752207, 0.04783897789654021], [0.7677013102662849, -0.043100364265406484, -0.36849626514810346, 0.30754213248167933, 0.004959008094369536, 0.029308575876387646, 0.04357265305699324, 0.1465919344971403, -0.05589856477778289, 0.0033358385986845974, 0.004424158382899468, 0.0044703229355311895, -0.026940426221490454, 0.020222233665937568, 0.5398712487987771, -1.3596020284018464, -1.2784436511711756, -0.7520592536795536, -0.04202277658478364, 0.09586228727610765], [0.005546368694395509, 0.17687012505351465, -0.040023217795586714, -0.19327435073721355, 0.147554571568895, 0.04357265305699324, -0.7857069470887235, 0.10007876322859674, -0.059680550692583, -0.0785846547632956, -0.04238889912221784, 0.1019796584895192, -0.0830762301096638, 0.012449489597615507, -0.006896607199701421, 0.05501284314612028, 0.09739072884066273, 0.14895440267728088, -0.008469897399518395, 0.37537866356052235], [-0.0729286030117644, 0.12544376772804366, 0.05859324640108397, -0.0017645082161845335, -0.22318989551544333, 0.1465919344971403, 0.10007876322859674, -0.036255916856893944, -0.010606811704753473, 0.01664372652317333, 0.059524323215301114, -0.042654527484456264, 0.07278484477468214, 0.09677771285690029, 0.015044986578989686, -0.03305123979286355, -0.028331258536381573, -0.024192586758357573, -0.005884652115888826, -0.08862491949845826], [0.37397354615613815, 0.0181253535871363, -0.09114539113414645, -0.0069501138033697014, -0.004866175346369206, -0.05589856477778289, -0.059680550692583, -0.010606811704753473, -0.014950406136544915, 0.006176093018373458, 0.003784659467510002, -0.020291597948714105, -0.005621417292641111, 0.036920383024897696, -0.0892924431803667, -0.4976884422270433, -0.1293767584261564, 0.07240820895553167, -0.04262105422660473, 0.003628799370577314], [-0.08042620082195048, 0.010153344712730047, 0.23031401027628223, -0.047714669042042045, 0.0008987428749639296, 0.0033358385986845974, -0.0785846547632956, 0.01664372652317333, 0.006176093018373458, 0.009032461381039015, -0.0021075296203598294, -0.0230393569024925, -0.003426206472382158, -0.0002028715358422294, -0.7950379948222186, 0.7992529975926908, 0.2628730467610331, -0.35072300821470387, -0.003370467874640551, 0.024608524596138652], [-0.7890830019183414, -0.015401960775276346, 0.23272173782282862, 0.07204534217944254, 0.014561223366043909, 0.004424158382899468, -0.04238889912221784, 0.059524323215301114, 0.003784659467510002, -0.0021075296203598294, 0.019391817421799663, -0.015131435832610951, 0.008439094270516534, -0.015882788153086906, 0.5447031651883806, -0.05528965453610969, -0.5862089614872267, -0.3432560533066222, -0.0009747993714340061, -0.003738260397534724], [0.8575548591940783, -0.0945495534121372, -0.16127433433163785, -0.06848418312183205, -0.005112150405225184, 0.0044703229355311895, 0.1019796584895192, -0.042654527484456264, -0.020291597948714105, -0.0230393569024925, -0.015131435832610951, 0.02338855664718642, 0.013856474882118977, -0.0032472482414809267, -1.0241584024536559, 0.9292769960328738, 0.3230467568590253, -0.3672653514084518, -0.0035222721376188724, -0.19560196684275483], [0.10714702369206586, -0.03337566055273211, -0.1929221177333412, 0.10564390936983957, -0.015786562371025447, -0.026940426221490454, -0.0830762301096638, 0.07278484477468214, -0.005621417292641111, -0.003426206472382158, 0.008439094270516534, 0.013856474882118977, -0.04039935231550901, 0.019029227397126418, 0.6360889667286818, -0.6364793578504124, -0.5825241030980205, -0.19689392288357144, 0.11785282840683235, 0.15437802929091754], [-0.9985761060440952, -0.19810140354734807, -0.07387062224176945, 0.05498202617474903, 0.019490267635763996, 0.020222233665937568, 0.012449489597615507, 0.09677771285690029, 0.036920383024897696, -0.0002028715358422294, -0.015882788153086906, -0.0032472482414809267, 0.019029227397126418, -0.021262936465021337, 1.7312998386432177, -0.28332303733415143, -0.20649101673218695, 0.23109552817428797, 0.07462888014196668, 0.360442811292237], [0.2071414045284976, -0.43489258065197106, 0.4803604601132522, 0.5938346065585351, -1.8007431311750377, 0.5398712487987771, -0.006896607199701421, 0.015044986578989686, -0.0892924431803667, -0.7950379948222186, 0.5447031651883806, -1.0241584024536559, 0.6360889667286818, 1.7312998386432177, 1.1501790807755135, 0.13325931350016793, -0.24200363874569636, -0.3738448520802398, -0.7066871405227658, 0.18859762966522758], [-0.03512783256485824, -0.8463129620807677, 0.23591221269865742, 0.08052089937825804, 0.3704914794264029, -1.3596020284018464, 0.05501284314612028, -0.03305123979286355, -0.4976884422270433, 0.7992529975926908, -0.05528965453610969, 0.9292769960328738, -0.6364793578504124, -0.28332303733415143, 0.13325931350016793, -0.7468809092669275, -0.6258200941447505, -0.5946982157956324, -0.6254427087897749, -0.3286385661511224], [-0.3074825987636097, -0.9168396051668359, 0.09527325521337521, -0.025130725893016986, 2.184304134923695, -1.2784436511711756, 0.09739072884066273, -0.028331258536381573, -0.1293767584261564, 0.2628730467610331, -0.5862089614872267, 0.3230467568590253, -0.5825241030980205, -0.20649101673218695, -0.24200363874569636, -0.6258200941447505, -1.3889853589196943, -0.5563152884671996, 0.2219302755626738, -0.6162098947219379], [-0.3916090733553413, -0.7116662481206637, -0.006357075362910785, 0.07199755053201044, 2.6608310954027474, -0.7520592536795536, 0.14895440267728088, -0.024192586758357573, 0.07240820895553167, -0.35072300821470387, -0.3432560533066222, -0.3672653514084518, -0.19689392288357144, 0.23109552817428797, -0.3738448520802398, -0.5946982157956324, -0.5563152884671996, -0.8026319285069571, 0.5227124645024093, -0.5916925709331989], [0.43341345647119317, -0.08455861051172718, 0.11026004010550705, -0.24382388077351724, 0.014948852708752207, -0.04202277658478364, -0.008469897399518395, -0.005884652115888826, -0.04262105422660473, -0.003370467874640551, -0.0009747993714340061, -0.0035222721376188724, 0.11785282840683235, 0.07462888014196668, -0.7066871405227658, -0.6254427087897749, 0.2219302755626738, 0.5227124645024093, -0.20112115387587368, -0.2410527363388057], [-0.1938256360252038, -0.007065635528035637, 0.14904182087895668, -0.13351492748106988, 0.04783897789654021, 0.09586228727610765, 0.37537866356052235, -0.08862491949845826, 0.003628799370577314, 0.024608524596138652, -0.003738260397534724, -0.19560196684275483, 0.15437802929091754, 0.360442811292237, 0.18859762966522758, -0.3286385661511224, -0.6162098947219379, -0.5916925709331989, -0.2410527363388057, 0.23891177800794236]],dtype=fnp.float32),fnp.asarray([-1.7897838591014175, -6.108930885283548, -0.44789257628106655, -4.424707039780121, -8.056087117284191, -3.3987191113481714, 0.8860835423292212, 4.529786328345346, -1.1697022794324579, -7.4451619303661145, -0.37949573407470594, 0.5047956469484879, 1.8414589773951207, 2.885968201634251, -10.219381303622765, 0.353843530326149, 3.1976634435373, 1.8912748378766597, -8.526006054519046, 3.4959115174725923, 2.4859127072835814, 1.8722618781665161, -0.9204785458184495, 3.6920940594974376, 0.34396787549204655, 2.990379737911606, 0.4683349987112566, -0.4442634359765856, 1.1563208289555142, 1.8692097288644733, 1.8320266349569105, -3.2553806608266624],dtype=fnp.float32),[-0.06701807971723067, 0.3536838780963392, -0.15085838513873134, -0.12664381302736508, 0.016603991664519598, 0.009770832707608913, -0.07960166360710214, -0.08731235764735153, 0.03702274095322178, 0.06402914706987878, -0.03255219981109399, 0.005991130060824891, -0.023625900367378255, 0.15624955955886713, -0.07234952729625393, -0.2845854280886076, -0.3395100841433624, 0.25757714028424117, 0.23296919177116515, -0.042670902755203494, 0.04049188326906676],[-0.4275534716260678, 0.11411407491844604, -0.1883262756117759, -0.05708877390640838, 0.0062005784035497995, 0.05461013730951119, -0.07285714763276095, -0.41432155695348405, 0.2012539324678321, 0.07361499376338633, -0.022896419941446373, -0.005905888466896539, -0.01523813455887655, 0.12454090643671814, -0.05836692001921941, -0.3028846520896411, 0.6222090451101723, -0.3921975539065682, -0.937202374456043, 0.07886399595034131, 0.12826765286735947],[-0.2968748751819268, 0.07073220891526777, 0.22464655254971727, 1.1716353752525759, 3.1193274144257344, -0.03463265813991678, 3.6582609905510357, -1.2901142088393698, 0.11141916460285514, 2.1903376020650813, -0.8112989959339186, -2.1188267228147697, 1.8057817191866312, -0.32922412129008305, -2.701860464979695, 0.1276118280995406, -0.5980023320266981, 2.2576057292983127, -0.6508440829429312, 1.4193325523627758, 0.9360720577150405, 2.293873227892505, 0.5174230923113189, 4.424528025982415, -0.24497216255150134, -0.6891983293027735, 0.053645290713709985, -4.8914432326560675, 0.4935516997992295, -0.8567152329129796, -0.23036667463571026, 0.5359617964930987],[-3.968073101475781, -1.4391793993897024, 0.18950306675109385, -1.1274713874136513, 0.8910540971073881, 1.215544575010965, -1.82798432083881, -0.8362239785413712, 0.13711555075814408, -3.708317949924944, -0.23941658555190626, 0.20967345110959823, -1.6505971964522939, 2.430929787218097, 1.664940500929582, 0.13929499322035738, 1.7266556730881273, -1.8183045440394345, -0.30194340084347765, -3.6837476159773574, -0.21089241737502418, 0.24072588835343733, 0.1750358829896111, 1.3404499374249286, -0.30493020386540376, 0.39380501594347317, 0.02196339513067711, 3.862641264583715, 0.4328656010439319, 3.8407671557681753, 4.634302259401812, -0.2911146440422683])

        if self.diagonal_transport_enabled:
            dt_beta=fnp.asarray(self.diagonal_transport_beta,dtype=fnp.float64)
            dt_q=fnp.zeros(fulln,dtype=fnp.float64)
            dt_d=fnp.zeros(fulln,dtype=fnp.float64)
            dt_corrections=[]
        if self.fourth_modulation:
            # Modulation arrays built once per distinct head (default, and the last-two-layer overrides).
            modulation_sets={}
            for mkey,(msm,mtm,msh,mth) in ((None,(self.shadow_modulation,self.tree_modulation,self.shadow_hidden_modulation,self.tree_hidden_modulation)),
                                           (0,head_override[0][4:8]),(1,head_override[1][4:8])):
                modulation_sets[mkey]=(fnp.asarray([msm[0],mtm[0]],dtype=fnp.float32),
                                       fnp.asarray([list(r_) for r_ in zip(msm[1:],mtm[1:])],dtype=fnp.float32),
                                       fnp.asarray([list(r_) for r_ in zip(msh,mth)],dtype=fnp.float32))
        if self.diagonal_transport_enabled:
            dt_mean_beta,dt_mean_bias=dt_beta[:55],dt_beta[55]
            dt_var_beta,dt_var_bias=dt_beta[56:111],dt_beta[111]
        for layer,fullw in enumerate(weights):
            layer_cutoff=self.cutoff
            compression_limit=self.compress_age if layer<mlp.depth-2 else mlp.depth+1
            bm,bc=pm,pc
            m=pm @ fullw
            pc=flops.as_symmetric(pc,symmetry=(0,1))
            if layer==0:
                c=fnp.einsum('ia,ib->ab',fullw,fullw)
            else:
                cov_active=fnp.nonzero(al > -3.25)[0]
                if len(cov_active)<fulln:
                    # Deep-tail units: keep only their variance on the diagonal.
                    cov_tail=fnp.nonzero(al <= -3.25)[0]
                    ka=len(cov_active);pad=(-ka)%64
                    wa=fullw[cov_active,:]
                    pca=pc[cov_active,:][:,cov_active]
                    if pad:
                        wa=fnp.pad(wa,((0,pad),(0,0)))
                        pca=fnp.pad(pca,((0,pad),(0,pad)))
                    cw=matmul(pca,wa,depth=5,upgrade=False)
                    c=symmetric_covariance_product(wa,cw)
                    wd=fullw[cov_tail,:]
                    c=c+fnp.diag(fnp.einsum('i,ij,ij->j',fnp.diag(pc)[cov_tail],wd,wd))
                else:
                    cw=matmul(pc,fullw,depth=5,upgrade=False)
                    c=symmetric_covariance_product(fullw,cw)
            v=fnp.maximum(fnp.diag(c),1e-9)
            sd=fnp.sqrt(v)
            al=m/sd
            pdf=flops.stats.norm.pdf(al).astype(fnp.float32)
            gain=flops.stats.norm.cdf(al).astype(fnp.float32)
            pm=m*gain+sd*pdf
            pm_raw2=m*pm+v*gain
            pv=fnp.maximum(pm_raw2-pm*pm,1e-12)
            pc_g2=pdf/sd
            pc=_fill(fnp.einsum('i,ij,j->ij',gain,c,gain)+fnp.einsum('i,ij,ij,j->ij',0.5*pc_g2,c,c,pc_g2),pv)
            pre_m,pre_c=m,c
            keep=fnp.nonzero(al > -layer_cutoff)[0] if layer<mlp.depth-1 else fnp.nonzero((al > -3.5)&(al < 3.5))[0]
            previous_weights=fullw[previous,:]
            w=previous_weights[:,keep]
            wt=w.T
            # C1b: every history block of the layer is column-packed once; the per-element math is unchanged.
            last=layer==mlp.depth-1
            fresh_n=len(fresh_pivots)
            def _special(f):
                if f[0] is not None or last:return False
                a=f[-1]+1
                if a>=compression_limit:return True
                return bool(self.aged_energy_tolerance) and (a>=2 if self.aged_every_age else a==2)
            q_order=[i for i,f in enumerate(factors) if f[0] is not None]
            none_all=[i for i,f in enumerate(factors) if f[0] is None]
            fresh_blocks=[i for i in none_all if fresh_n and factors[i][-1]==0]
            none_order=([i for i in none_all if i not in fresh_blocks and _special(factors[i])]
                        +[i for i in none_all if i not in fresh_blocks and not _special(factors[i])]+fresh_blocks)
            block_order=none_order+q_order
            parts=[];xold={};zcol={};qcol={};xseg={};col=0
            for i in none_order:
                fxr=factors[i][1];fr=fresh_n if i in fresh_blocks else 0;old=fxr.shape[1]-fr
                if old:parts.append(fxr[:,:old] if fr else fxr)
                xold[i]=(col,old,fr);col+=old
            ystart=col
            for i in none_order:
                if factors[i][2].shape[1]:parts.append(factors[i][2])
                col+=factors[i][2].shape[1]
            yend=zstart=col
            for i in block_order:
                r=factors[i][6].shape[1]
                if r:parts.append(factors[i][6])
                zcol[i]=(col,r);col+=r
            zend=col
            for i in q_order:
                r=factors[i][0].shape[1]
                if r:parts.append(factors[i][0])
                qcol[i]=(col,r);col+=r
            total_cols=col
            col=0
            for i in block_order:
                xseg[i]=(col,col+factors[i][1].shape[1]);col=xseg[i][1]
            kall=col
            knone=xseg[none_order[-1]][1] if none_order else 0
            packed_cat=[]
            def _packed():
                if not packed_cat:packed_cat.append(_cat(parts,1))
                return packed_cat[0]
            cm=_cat([factors[i][8] for i in block_order],1) if factors else None
            cmr=fnp.split(cm,cm.shape[0],axis=0) if factors else None
            Xk=Yk=Ek=Zk_all=None
            zk={};qk={};joint_z=None
            def _zk(i):
                if i not in zk:zk[i]=joint_z[0][i][joint_z[1]:]
                return zk[i]
            radial3=3*radial
            full_g=False
            band_rows=None
            band_btd=None
            md_full=fnp.einsum('ij,ij->j',fullw,fullw)
            if layer<mlp.depth-1:
                band=fnp.nonzero((al > -self.band_limit)&(al <= -layer_cutoff))[0]
                if len(band):
                    # Gaussian derivative ratios on the full width once, then gathered for band and kept units.
                    full_g=True
                    g3_full=-al*pdf/v
                    al2_full=al*al
                    g4_full=(al2_full-1)*pdf/(v*sd)
                    nb=len(band)
                    bw=previous_weights[:,band]
                    bwt=bw.T
                    if factors:
                        joint=False
                        band_projection=None
                        if parts and total_cols*bw.shape[0]<=26000000 and total_cols*nb<=26000000:
                            if joint_projection_profitable(nb,len(keep),bw.shape[0],total_cols):
                                joint_rows=fnp.concatenate((bwt,wt),axis=0)
                                joined_projection=matmul(joint_rows,_packed(),depth=4)
                                joint=True
                            else:band_projection=bwt @ _packed()
                        if joint:
                            jx,jy,jz,jq=_assemble_blocks(joined_projection,joint_rows,factors,none_order,q_order,xold,zcol,qcol,ystart,yend,fresh_pivots,False,qpair)
                            je=_cat([jz[i] @ factors[i][9] for i in block_order],1)
                            # band rows stay in the joint arrays; their td comes out of the main contraction
                            band_rows=(jx,jy,je)
                            Xk,Yk=jx[nb:],jy[nb:]
                            joint_z=(jz,nb)
                            qk={i:jq[i][nb:] for i in q_order}
                            if zend>zstart:Zk_all=joined_projection[nb:,zstart:zend]
                        else:
                            bx,by,bz,_=_assemble_blocks(band_projection,bwt,factors,none_order,q_order,xold,zcol,qcol,ystart,yend,fresh_pivots,False,lambda bq,fxr,fyr:(bq @ fxr,bq @ fyr))
                            br=_cat([bz[i] @ factors[i][9] for i in block_order],1)
                            bx2=bx*bx
                            band_btd=(fnp.einsum('ij,ij,ij,kj->i',bx,by,by,cmr[4])+fnp.einsum('ij,ij,kj->i',by,bx2,cmr[1])
                                      +fnp.einsum('ij,ij,kj->i',bx2,bx,cmr[15])+fnp.einsum('ij,ij->i',br,bx2))
                    else:band_btd=fnp.zeros(nb,dtype=fnp.float32)
                    if band_btd is not None:
                        pm,pc=_restore_band(band,band_btd,pc_g2,g3_full,g4_full,md_full[band],radial3,pm,pm_raw2,pc,fulln)

            full_v,full_sd,full_pdf=v,sd,pdf
            n=len(keep)
            approximate_compression_repair=self.tail_repair_mode<2 and layer>=self.tail_repair_start and n>self.tail_rank
            eye=fnp.eye(n,dtype=fnp.float32)
            m=pre_m[keep]
            c=pre_c[keep,:][:,keep]
            if layer==0:
                metric=c
                md=fnp.diag(metric)
                k22=radial*(fnp.outer(md,md)+2*metric*metric)
            else:
                md=md_full[keep]
                md2=md*md
                kd=radial3*md2
                k22=_fill(radial*(1+2/fulln)*fnp.outer(md,md),kd)
            if layer==0:kd=fnp.diag(k22)
            if layer==0:md2=md*md
            v=full_v[keep]
            sd=full_sd[keep]
            alpha=al[keep]
            pdf=full_pdf[keep]
            gaussian_p=gain[keep]
            p=gaussian_p
            gaussian_g2=pdf/sd
            g2=gaussian_g2
            if full_g:
                g3=g3_full[keep]
                alpha2=al2_full[keep]
                g4=g4_full[keep]
            else:
                g3=-alpha*pdf/v
                alpha2=alpha*alpha
                g4=(alpha2-1)*pdf/(v*sd)
            # m*p+sd*pdf and m*raw1+v*p on the kept units are exactly the full-width pm and pm_raw2 (band units are disjoint).
            raw1=pm[keep]
            raw2=pm_raw2[keep]
            raw3=m*raw2+2*v*raw1
            raw4=m*raw3+3*v*raw2
            gaussian_kurtosis=raw4-4*raw1*raw3+6*raw1*raw1*raw2-3*raw1**4-3*(raw2-raw1*raw1)**2
            t=None
            abs_alpha=fnp.abs(alpha)
            readout_rows=fnp.nonzero(abs_alpha<self.readout_cutoff)[0] if self.readout_cutoff<100 and layer<mlp.depth-1 else None
            sparse_parts=None
            projected=[]
            packed_left=[]
            packed_right=[]
            packed_left_c=[]
            packed_right_c=[]
            missing_c=[]
            ft_diag=None
            packed_residual=[]
            packed_z=[]
            readout_cols=sum((2*f[1].shape[1] if f[0] is None else f[0].shape[1]) for f in factors if f[0] is not None or f[-1]+1<compression_limit or approximate_compression_repair)
            combine_readout=readout_cols*n<=26000000
            if not factors:
                shadow_tree=fnp.zeros((2,n),dtype=fnp.float32)
                td=fnp.zeros(n,dtype=fnp.float32)
            else:
                if Xk is None:
                    packed_projection=None
                    if parts and total_cols*w.shape[0]<=26000000 and total_cols*n<=26000000:
                        packed_projection=matmul(wt,_packed(),depth=4)
                    Xk,Yk,zk,qk=_assemble_blocks(packed_projection,wt,factors,none_order,q_order,xold,zcol,qcol,ystart,yend,fresh_pivots,True,qpair)
                    Ek=_cat([zk[i] @ factors[i][9] for i in block_order],1)
                    if packed_projection is not None and zend>zstart:Zk_all=packed_projection[:,zstart:zend]
                if band_rows is not None:MX,MY,ME=band_rows;roff=nb
                else:MX,MY,ME=Xk,Yk,Ek;roff=0
                _rc=(lambda a_,seg_:a_[roff:,seg_[0]:seg_[1]]) if roff else (lambda a_,seg_:_cols(a_,seg_,kall))
                xy=MX*MY
                x2=MX*MX
                y2=MY*MY
                x4=x2*x2
                # rows: shadow = sum(a4*xy*y2 + c4*x4), tree = sum(b4*x2*y2 + d4*x4) over the packed columns
                shadow_tree=fnp.einsum('ij,kj->ki',x4,cm[7:12:4])+fnp.stack((fnp.einsum('ij,ij,kj->i',xy,y2,cmr[6]),fnp.einsum('ij,ij,kj->i',x2,y2,cmr[14])))
                if roff:shadow_tree=shadow_tree[:,roff:]
                if last:
                    td=fnp.einsum('ij,ij,j->i',xy,Yk,cm[4])+fnp.einsum('ij,ij,j->i',xy,Xk,cm[1])+fnp.einsum('ij,ij,j->i',x2,Xk,cm[15])+fnp.einsum('ij,ij->i',Ek,x2)
                else:
                    # Same contraction as einsum('ik,ijk->jk',mix,stack(terms)) (identical billed FLOPs), written
                    # elementwise on the packed columns so no 4-plane stack is materialised.
                    ex=ME*MX
                    left=cmr[0]*xy+cmr[1]*x2
                    right=cmr[2]*xy+cmr[3]*x2+cmr[4]*y2+cmr[5]*ex
                    if readout_rows is not None or roff:
                        # diag of the readout; on band rows this equals ac*x*y^2+bc*x^2*y+cc*x^3+e*x^2
                        td=(fnp.einsum('ij,ij->i',left,MY)+fnp.einsum('ij,ij->i',right,MX)+fnp.einsum('ij,ij->i',ME,x2))/3
                        if roff:
                            pm,pc=_restore_band(band,td[:roff],pc_g2,g3_full,g4_full,md_full[band],radial3,pm,pm_raw2,pc,fulln)
                            td=td[roff:]
                    if readout_rows is None:td=fnp.zeros(n,dtype=fnp.float32)
                    packable=lambda i:factors[i][0] is None and (factors[i][-1]+1<compression_limit or approximate_compression_repair) and combine_readout
                    packed_none=[i for i in none_order if packable(i)]
                    if packed_none and len(packed_none)==len(none_order):
                        if q_order:pl,pr,py,px=left[roff:,:knone],right[roff:,:knone],Yk[:,:knone],Xk[:,:knone]
                        elif roff:pl,pr,py,px=left[roff:],right[roff:],Yk,Xk
                        else:pl,pr,py,px=left,right,Yk,Xk
                        packed_left.extend((pl,pr))
                        packed_right.extend((py.T,px.T))
                    else:
                        for i in packed_none:
                            packed_left.extend((_rc(left,xseg[i]),_rc(right,xseg[i])))
                            packed_right.extend((_cols(Yk,xseg[i],kall).T,_cols(Xk,xseg[i],kall).T))
                    fts={}
                    for i in none_order:
                        if not packable(i):
                            lr=fnp.concatenate((_rc(left,xseg[i]),_rc(right,xseg[i])),axis=1)
                            yx=fnp.concatenate((_cols(Yk,xseg[i],kall).T,_cols(Xk,xseg[i],kall).T),axis=0)
                            fts[i]=(matmul(lr,yx,depth=4)+(_rc(x2,xseg[i]) @ factors[i][7]) @ _zk(i).T)/3
                    for i in q_order:
                        f=factors[i]
                        li,ri=_rc(left,xseg[i]),_rc(right,xseg[i])
                        if combine_readout:
                            if readout_rows is None:
                                packed_left_c.append(paired_readout(li,ri,f[1],f[2]))
                            else:
                                lrc=fnp.concatenate((li,ri),axis=1)
                                ryx=qryx(f[1],f[2])
                                packed_left_c.append(compact_product(lrc[readout_rows,:],ryx))
                                missing_c.append((lrc,ryx))
                            packed_right_c.append(qk[i].T)
                        else:
                            fts[i]=((paired_readout(li,ri,f[1],f[2])) @ qk[i].T+(_rc(x2,xseg[i]) @ f[7]) @ _zk(i).T)/3
                    packed_ids=[i for i in block_order if i not in fts]
                    for i in packed_ids:packed_residual.append(_rc(x2,xseg[i]) @ factors[i][7])
                    if packed_ids:
                        if len(packed_ids)==len(block_order) and Zk_all is not None:packed_z.append(Zk_all.T)
                        else:packed_z.extend(_zk(i).T for i in packed_ids)
                    for i in range(len(factors)):
                        ft=fts.get(i)
                        if ft is not None:
                            t=ft if t is None else t+ft
                            if readout_rows is not None:ft_diag=fnp.diag(ft) if ft_diag is None else ft_diag+fnp.diag(ft)
                        projected.append((i,ft))
            if packed_left or packed_left_c:
                if readout_rows is None:
                    readout_t=(matmul(fnp.concatenate(packed_left+packed_left_c,axis=1),fnp.concatenate(packed_right+packed_right_c,axis=0),depth=4)+_cat(packed_residual,1) @ _cat(packed_z,0))/3
                    t=readout_t if t is None else t+readout_t
                else:
                    sly=fnp.concatenate(packed_left,axis=1) if packed_left else None
                    sr=fnp.concatenate(packed_right+packed_right_c,axis=0)
                    rl=_cat(packed_residual,1)
                    rr=_cat(packed_z,0)
                    parts=([sly[readout_rows,:]] if packed_left else [])+packed_left_c
                    slr=fnp.concatenate(parts,axis=1) if len(parts)>1 else parts[0]
                    small=(matmul(slr,sr,depth=4)+rl[readout_rows,:] @ rr)/3
                    # Place the computed rows: row r of the result is small[k] if r is the k-th readout row, else 0.
                    readout_count=fnp.bincount(readout_rows,minlength=n)
                    placed=fnp.take(fnp.concatenate((fnp.zeros((1,n),dtype=small.dtype),small)),fnp.cumsum(readout_count)*readout_count,axis=0)
                    t=placed if t is None else t+placed
                    packed_diagonal=td if ft_diag is None else td-ft_diag
                    outside=1-readout_count.astype(fnp.float32)
                    sparse_parts=(sly,missing_c,sr,rl,rr,packed_diagonal,outside)
            if t is None:t=fnp.zeros((n,n),dtype=fnp.float32)
            if layer<mlp.depth-1:
                if readout_rows is None:td=fnp.diag(t)
                else:t=_fill(t,td)
            mu=raw1+td*(g3*(self.s_mu_td/6))+kd*(g4*(self.s_mu_kd/24))
            raw2=raw2+td*(g2*(self.s_var_td/3))+kd*(g3*(self.s_var_kd/12))
            raw3=raw3+td*p+kd*g2/4
            raw4=raw4+4*td*raw1+kd*p
            mu2=mu*mu
            var=fnp.maximum(raw2-mu2,1e-9)
            row=pm+fnp.bincount(keep,weights=mu-raw1,minlength=fulln).astype(fnp.float32)
            norm=fnp.maximum(v,1e-8)
            standardized_mean=m/sd
            c_squared=c*c
            local_num=fnp.stack((alpha,sd,p,pdf,td,kd,alpha,alpha,
                fnp.sum(c_squared,axis=1),fnp.einsum('ij,ij->i',c_squared,c),fnp.einsum('ij,ij->i',t,c),
                c @ standardized_mean,t @ standardized_mean,fnp.einsum('ij,ij->i',t,t),
                mu,raw2,raw3,raw4,fnp.full_like(alpha,layer/16),fnp.broadcast_to(radial,alpha.shape)),axis=1)
            local_scale=((1/norm)[:,None]**local_pinv)*((1/sd)[:,None]**local_psd)*local_const
            local=local_num*local_scale
            standardized=fnp.clip((local-feature_mean)/feature_scale,-5,5)
            hb,hl,hq,h4,hsm,htm,hsh,hth=head_override.get(mlp.depth-1-layer,(feature_bias,feature_linear,feature_quadratic,neural_4,self.shadow_modulation,self.tree_modulation,self.shadow_hidden_modulation,self.tree_hidden_modulation))
            response=hb+standardized @ hl+fnp.einsum('ij,jk,ik->i',standardized,hq,standardized)
            neural_hidden=fnp.tanh(standardized @ neural_0+neural_1)
            neural_hidden=fnp.tanh(neural_hidden @ neural_2+neural_3)
            response=response+(neural_hidden @ h4)  # neural_5 is the constant 0.0
            source=response*pdf*sd/fulln
            # Conditional shadow/tree sources as rows of one (2, n) array.
            conditional=(shadow_tree-radial3c*md2)*(g4/24)
            if self.fourth_modulation:
                mod_bias,mod_slopes,mod_hidden=modulation_sets[mlp.depth-1-layer if mlp.depth-1-layer in head_override else None]
                modulation=standardized @ mod_slopes+mod_bias+neural_hidden @ mod_hidden
                source=source+fnp.einsum('ij,ji->i',modulation,conditional)
            gains=gain
            channel_source=fnp.vstack((source,conditional))
            channel_ids=(channel_offsets+keep).reshape(-1)
            channel_scatter=fnp.bincount(channel_ids,weights=channel_source.reshape(-1),minlength=3*fulln).astype(fnp.float32).reshape((3,fulln))
            channel_mean=(channel_mean @ fullw)*gains+channel_scatter
            channel_history.append(channel_mean)
            if self.mean_translation_fraction:
                mt_delta=channel_weights @ channel_scatter
                row=row+mt_delta*(self.mean_translation_fraction/(1-4/(4*fulln)+4/(32*fulln*fulln)+20/(128*fulln*fulln*fulln)))
            if self.diagonal_transport_enabled:
                dt_sd=sd.astype(fnp.float64)
                dt_pdf=pdf.astype(fnp.float64)
                dt_probability=gain.astype(fnp.float64)
                # standardized and neural_hidden are float64 already (float64 feature constants): no casts needed.
                dt_local=fnp.concatenate((standardized,neural_hidden,
                    (channel_mean[:,keep].astype(fnp.float64)*fulln/dt_sd).T),axis=1)
                if mlp.depth-3<=layer<mlp.depth-1:
                    _cr24[layer]=(keep,standardized[:,0],standardized[:,4],standardized[:,5],dt_pdf,dt_sd,dt_probability)
                if layer==mlp.depth-1 and mlp.depth==16 and fulln==1024 and (mlp.depth-2) in _cr24 and (mlp.depth-3) in _cr24:
                    _z=standardized
                    _h=neural_hidden
                    _ps=(dt_pdf*dt_sd/fulln)[:,None]
                    _cst=conditional.T.astype(fnp.float64)
                    _cs=_cst[:,0:1]
                    _ct=_cst[:,1:2]
                    _one=fnp.ones((len(keep),1),dtype=fnp.float64)
                    _iu=(fnp.asarray([_i for _i in range(20) for _j in range(_i,20)]),fnp.asarray([_j for _i in range(20) for _j in range(_i,20)]))
                    _phi=fnp.concatenate((_one,_z,_z[:,_iu[0]]*_z[:,_iu[1]],_h),axis=1)
                    _psi=fnp.concatenate((_one,_z,_h),axis=1)
                    _A=fnp.concatenate((_ps*_phi,_cs*_psi,_ct*_psi),axis=1)
                    _W15=fullw.astype(fnp.float64)
                    _W15s=_W15*_W15
                    _W14=weights[mlp.depth-2].astype(fnp.float64)
                    _W14s=_W14*_W14
                    _M=[]
                    for _L,_mats in ((mlp.depth-2,None),(mlp.depth-3,(_W14,_W14s))):
                        _k,_a,_t,_kd,_pf,_sdv,_pr=_cr24[_L]
                        for _v in (_pf,_sdv,_a,_t,_kd,None):
                            _full=_pr*(1-_pr) if _v is None else fnp.bincount(_k,weights=_v,minlength=fulln)
                            if _mats is None:_M+=[_full @ _W15,_full @ _W15s]
                            else:_M+=[(_full @ _mats[0]) @ _W15,(_full @ _mats[1]) @ _W15s]
                    _M=fnp.stack(_M,axis=1)[keep]
                    _Mm=fnp.sum(_M,axis=0)/len(keep)
                    _Md=_M-_Mm
                    _Ms=fnp.sqrt(fnp.sum(_Md*_Md,axis=0)/len(keep))
                    _M=_Md/(_Ms+1e-12)
                    _B=fnp.concatenate((_A,_ps*_M,_ps*_M*_z[:,:1],_ps*_M*_z[:,1:2]),axis=1)
                    _cr24_extra=fnp.bincount(keep,weights=_B @ fnp.asarray(self.cr24_beta,dtype=fnp.float64),minlength=fulln)
                dt_source_mean=dt_local @ dt_mean_beta+dt_mean_bias
                dt_source_variance=dt_local @ dt_var_beta+dt_var_bias
                # float64 products with the float32 weights (numpy promotes; no float64 weight copy).
                dt_a=dt_q @ fullw
                dt_b=fnp.einsum('i,ij,ij->j',dt_d,fullw,fullw)
                dt_ratio=fnp.bincount(keep,weights=gaussian_g2,minlength=fulln)
                if self.mean_translation_fraction:
                    dt_parent_mean=(row*(1-4/(4*fulln)+4/(32*fulln*fulln)+20/(128*fulln*fulln*fulln))+(1-self.mean_translation_fraction)*(channel_weights @ channel_mean)).astype(fnp.float64)
                else:
                    dt_parent_mean=(row*(1-4/(4*fulln)+4/(32*fulln*fulln)+20/(128*fulln*fulln*fulln))+channel_weights @ channel_mean).astype(fnp.float64)
                dt_q=dt_probability*dt_a+(.5*dt_ratio)*dt_b+fnp.bincount(keep,weights=dt_sd*dt_pdf*dt_source_mean,minlength=fulln)
                dt_d=(2*dt_parent_mean*(1-dt_probability))*dt_a+(dt_probability-dt_parent_mean*dt_ratio)*dt_b+fnp.bincount(keep,weights=(dt_sd*dt_sd)*dt_source_variance,minlength=fulln)
                dt_corrections.append(dt_q)
            rows.append(row)
            if layer==mlp.depth-1:
                break
            closure_z=fnp.clip(alpha,-3.0,3.0)/3.0
            pdf_v2=pdf/(v*v)
            g5=alpha*(3-alpha2)*pdf_v2
            g6=(alpha2*(alpha2-6)+3)*pdf_v2/sd
            kd24=kd/24
            p=fnp.clip(p+td*(g4*(self.s_pg_td/6))+kd24*g5,0,1)
            g2=fnp.maximum(g2+td*(g5*(self.s_pg_td/6))+kd24*g6,0)
            uu_rows=fnp.split(uu_c0+closure_z*(uu_a1+uu_a2*closure_z),len(uu_index),axis=0) if uu_index else []
            uu_row=lambda k:uu_rows[uu_index[k]] if k in uu_index else None
            if self.fused_star_closures:
                cg2=gaussian_g2
                cq2=2*(gaussian_p-mu*cg2)
                cq3=2*(cg2-mu*g3)
                fused_mixed,fused_cubic=fused_gaussian_star_closures(c,t,cg2,g3,cq2,cq3,c_squared)
            pair=0.5*c_squared+k22*(0.25*self.s_pair_k22)
            # X*outer(u,u) and X*outer(a,b) products are contracted directly (no outer/half_t temporaries).
            u_pair=uu_row(0)
            if u_pair is None:pair_term=fnp.einsum('i,ij,j->ij',g2,pair,g2)
            else:
                g2u=g2*u_pair
                pair_term=fnp.einsum('ki,ij,kj->ij',g2u,pair,g2u)
            cov=fnp.einsum('i,ij,j->ij',p,c,p)+pair_term
            u_corr=uu_row(1)
            corr_left=g2*(0.5*self.s_corr) if u_corr is None else (g2*u_corr)*(0.5*self.s_corr)
            corr_right=p if u_corr is None else p*u_corr
            corr=fnp.einsum('ki,ij,kj->ij',corr_left,t,corr_right) if u_corr is not None else fnp.einsum('i,ij,j->ij',corr_left,t,corr_right)
            cov=cov+corr+corr.T
            if self.fused_star_closures:
                for plane_,k_,s_ in ((fused_mixed[0],2,self.s_fmix),(fused_cubic[0],3,self.s_fcub)):
                    u_=uu_row(k_)
                    cov=cov+(_sc(plane_,s_) if u_ is None else fnp.einsum('ki,ij,kj->ij',u_*s_,plane_,u_))
            else:
                if self.correlation_third_order:
                    cg2=gaussian_g2
                    cov=cov+correlation_mixed_third(c,t,cg2,g3,g4,cg2,g3,g4,self.correlation_third_order)
                if self.gaussian_star_pairs:
                    gaussian_c3=c*c*c/6
                    cov=cov+gaussian_c3*fnp.outer(g3,g3)
            cov=_fill(cov,var)
            half_q1=mu*(1-p)
            half_q2=p-mu*g2
            q1=2*half_q1
            q2=2*half_q2
            desired=(fnp.einsum('ij,i,j->ij',c,q1,p)+fnp.einsum('ij,i,j->ij',pair,q2,g2)
                     +fnp.einsum('ij,i,j->ij',t,half_q2,p)+fnp.einsum('ji,i,j->ij',t,half_q1,g2))
            if self.fused_star_closures:
                fused_high=_sc(fused_mixed[1:]+fused_cubic[1:],self.s_fhigh)
                desired=desired+fused_high[0]
            else:
                if self.correlation_third_order and self.correlation_third_all:
                    cq2=2*(gaussian_p-mu*cg2)
                    cq3=2*(cg2-mu*g3)
                    cq4=2*(g3-mu*g4)
                    desired=desired+correlation_mixed_third(c,t,cq2,cq3,cq4,cg2,g3,g4,self.correlation_third_order)
                if self.gaussian_star_pairs:
                    desired=desired+gaussian_c3*fnp.outer(cq3,g3)
            diag3=raw3-3*mu*raw2+(2*mu2)*mu
            desired=_fill(desired,diag3)
            # Gaussian/Edgeworth covariance of centered squares, then project K4
            # onto the rotationally invariant tensor sym(I x I).
            kpost=fnp.einsum('ij,i,j->ij',c,q1,q1)+fnp.einsum('ij,i,j->ij',pair,q2,q2)-2*cov*cov
            skewterm=fnp.einsum('ij,i,j->ij',t,half_q2,q1)
            kpost=kpost+skewterm+skewterm.T
            if self.fused_star_closures:
                kpost=kpost+fused_high[1]
            else:
                if self.correlation_third_order and self.correlation_third_all:
                    kpost=kpost+correlation_mixed_third(c,t,cq2,cq3,cq4,cq2,cq3,cq4,self.correlation_third_order)
                if self.gaussian_star_pairs:
                    kpost=kpost+gaussian_c3*fnp.outer(cq3,cq3)
            diag4=raw4-4*mu*raw3+6*mu2*raw2-3*(mu2*mu2)-3*var*var
            kpost=_fill(kpost,diag4)
            radial_numerator=fnp.sum(kpost)
            if sparse_parts is not None:
                sly,missing_c,sr,rl,rr,packed_diagonal,outside=sparse_parts
                qout=q2*outside
                qs=[qout @ sly] if sly is not None else []
                for lrc,ryx in missing_c:qs.append((qout @ lrc) @ ryx)
                qsl=fnp.concatenate(qs) if len(qs)>1 else qs[0]
                exact_missing=(qsl @ (sr @ q1)+(qout @ rl) @ (rr @ q1))/3
                exact_missing=exact_missing-fnp.einsum('i,i,i->',qout,q1,packed_diagonal)
                radial_numerator=radial_numerator+exact_missing
            radial=radial_numerator*((1+self.radial_adjustments[min(layer//3,len(self.radial_adjustments)-1)])/(fulln*(fulln+2)))
            old_factors=factors
            factors=[]
            trace_parts=[]
            tail_repair=None
            run_ids=set()
            zp_all=fnp.einsum('ij,i->ij',Zk_all,p) if (projected and joint_z is not None and Zk_all is not None) else None
            if projected:
                xp_all=fnp.einsum('ij,i->ij',Xk,p)
                yp_all=fnp.einsum('ij,i->ij',Yk,p)
                start=kall
                for i in reversed(block_order):
                    if _special(old_factors[i]):break
                    run_ids.add(i);start=xseg[i][0]
                if run_ids and start<kall:
                    run_weights=trace_weights(cm)
                    if start:trace_parts.append(fourth_traces(xp_all[:,start:],yp_all[:,start:],run_weights[:,:,start:]))
                    else:trace_parts.append(fourth_traces(xp_all,yp_all,run_weights))
            for i,ft in projected:
                q,xr,yr,ac,bc,cc,_,vv,coeffs,vvt,age=old_factors[i]
                age=age+1
                mix=coeffs
                if zp_all is not None:
                    za_=zcol[i][0]-zstart
                    z=zp_all[:,za_:za_+zcol[i][1]]
                else:z=fnp.einsum('ij,i->ij',_zk(i),p)
                if q is not None:
                    if i not in run_ids:
                        trace_parts.append(fourth_traces(_cols(xp_all,xseg[i],kall),_cols(yp_all,xseg[i],kall),trace_weights(coeffs)))
                    factors.append((fnp.einsum('ij,i->ij',qk[i],p),xr,yr,ac,bc,cc,z,vv,coeffs,vvt,age))
                    continue
                x=_cols(xp_all,xseg[i],kall)
                y=_cols(yp_all,xseg[i],kall)
                if age<compression_limit:
                    if self.aged_energy_tolerance and (age>=2 if self.aged_every_age else age==2):
                        retained=aged_retained_columns(x,y,ac,bc,cc,z,vv,self.aged_energy_tolerance)
                        x=x[:,retained];y=y[:,retained]
                        ac=ac[retained];bc=bc[retained];cc=cc[retained]
                        vv=vv[retained,:]
                        vvt=vv.T
                        coeffs=coeffs[:,retained]
                    if i not in run_ids:
                        trace_parts.append(fourth_traces(x,y,trace_weights(coeffs)))
                    factors.append((None,x,y,ac,bc,cc,z,vv,coeffs,vvt,age))
                    continue
                xn=fnp.sum(x*x,axis=0)
                yn=fnp.sum(y*y,axis=0)
                cross=fnp.sqrt(xn*yn)
                wx=fnp.abs(ac)*yn+fnp.abs(bc)*cross+fnp.abs(cc)*xn
                wy=fnp.abs(ac)*cross+fnp.abs(bc)*xn
                sx=fnp.sqrt(wx)
                sy=fnp.sqrt(wy)
                joined=fnp.concatenate((x*sx,y*sy),axis=1)
                q,sv,vh=fnp.linalg.svd(joined,k=min(n,224 if layer<8 else self.tail_rank),full_matrices=False)
                if float(fnp.minimum(fnp.min(sx),fnp.min(sy)))>0:
                    svh=sv[:,None]*vh
                    kc=x.shape[1]
                    xr=svh[:,:kc]/sx
                    yr=svh[:,kc:]/sy
                else:xr,yr=paired_projection(q.T,x,y)
                if approximate_compression_repair:
                    repair_traces,diagonal_repair=trace_summary_history_repair(q,xr,yr,x,y,ac,bc,cc,z,vv,coeffs,self.trace_summary_strength)
                    trace_parts.append(repair_traces)
                    if diagonal_repair is not None:tail_repair=diagonal_repair if tail_repair is None else tail_repair+diagonal_repair
                else:
                    ca,cb=paired_projection(q,xr,yr)
                    trace_parts.append(fourth_traces(ca,cb,trace_weights(coeffs)))
                    extra=z @ vv.T
                    xy=ca*cb
                    x2=ca*ca
                    terms=fnp.stack((xy,x2,cb*cb,extra*ca))
                    left=fnp.einsum('ik,ijk->jk',mix[0:2],terms[:2])
                    right=fnp.einsum('ik,ijk->jk',mix[2:6],terms)
                    represented=((paired_readout(left,right,xr,yr)) @ q.T+(x2 @ vv) @ z.T)/3
                    tail_piece=ft*fnp.outer(p*p,p)-represented
                    tail_repair=tail_piece if tail_repair is None else tail_repair+tail_piece
                factors.append((q,xr,yr,ac,bc,cc,z,vv,coeffs,vvt,age))
            pp=p*p
            existing=_fill(fnp.einsum('ij,i,j->ij',t,pp,p),td*pp*p)
            alpha_sd=alpha/sd
            mixed_depth=0.5*self.calibrated_mixed_depth[min(layer,15)]
            b=fnp.einsum('i,ij->ij',p,c)+fnp.einsum('i,ij->ij',g2*mixed_depth,t)-fnp.einsum('i,ji,j->ij',p*mixed_depth,t,alpha_sd)
            if self.gaussian_star_strength:
                # gaussian_third_star_shift(c,gaussian_g2,-alpha/sd) with zero diagonal, scaled.
                b=b+_fill(fnp.einsum('i,ij,j->ij',gaussian_g2*(0.5*self.gaussian_star_strength),c_squared,-alpha_sd),0.0)
            b_squared=b*b
            db=fnp.diag(b)
            path=fnp.einsum('i,ji->ij',2*g2*db,b)+b_squared*g2
            pivots=fnp.nonzero(abs_alpha<self.kink_cutoff)[0]
            correction=desired-existing-path
            if tail_repair is not None:correction=correction+tail_repair
            da=fnp.diag(correction)
            a=_fill(3*correction.T,da)
            if len(pivots):
                b_norm=fnp.sum(b_squared,axis=0)
                db2=db*db
            if len(pivots) and layer>=2:
                dy=db
                ad=da
                ac_all=3*g2
                acdy2=ac_all*db2
                yn=fnp.maximum(b_norm-db2,0)
                mixed=a+b*(2*ac_all*dy)
                mixed_norm=fnp.maximum(fnp.einsum('ij,ij->j',mixed,mixed)-(ad+2*acdy2)**2,0)
                ac_yn=ac_all*yn
                energy=(acdy2+ad)**2+mixed_norm/3+ac_yn*ac_yn/3
                selected_energy=energy[pivots]
                order=fnp.argsort(selected_energy)
                cumulative=fnp.cumsum(selected_energy[order])
                pivots=pivots[fnp.sort(order[cumulative>=self.birth_energy_tolerance*fnp.sum(selected_energy)])]
            if len(pivots):
                bc=(fnp.einsum('ij,ij->j',a,b)-da*db)/fnp.maximum(b_norm-db2,1e-12)
                cc=da-db*bc
                x=eye[:,pivots]
                y=b[:,pivots]
                bc=bc[pivots]
                cc=cc[pivots]
                remainder=a[:,pivots]-y*bc-x*cc
                u,sv,vh=fnp.linalg.svd(remainder,k=min(n,len(pivots),self.repair_rank if layer<8 else 2),full_matrices=False)
                z=u*sv
                vv=vh.T
                # Every source is a selected coordinate column of x.
                cc=cc-fnp.einsum('ij,ij->i',z[pivots,:],vv)
                g2p=g2[pivots]
                dbp=db[pivots]
                ac=3*g2p
                a4=4*g3[pivots]
                c4=gaussian_kurtosis[pivots]-a4*dbp**3
                b4=12*g2p*g2p*v[pivots]
                d4=-b4*dbp*dbp
                birth_coeffs=coefficient_views(ac,bc,cc,a4,c4,b4,d4,coef_scale)
                # x = eye[:,pivots]: column norms are exactly 1 and x.y is exactly dbp.
                trace_parts.append(identity_fourth_traces(y,dbp,trace_weights(birth_coeffs)))
                if layer%2 and factors and factors[-1][0] is None and factors[-1][-1]<2:
                    _,ox,oy,oa,ob,oc,oz,ov,old_coeffs,_,_=factors[-1]
                    blockv=fnp.concatenate((fnp.pad(ov,((0,0),(0,vv.shape[1]))),fnp.pad(vv,((0,0),(ov.shape[1],0)))),axis=0)
                    ac=fnp.concatenate((oa,ac));bc=fnp.concatenate((ob,bc));cc=fnp.concatenate((oc,cc))
                    factors[-1]=(None,fnp.concatenate((ox,x),axis=1),fnp.concatenate((oy,y),axis=1),
                                  ac,bc,cc,fnp.concatenate((oz,z),axis=1),blockv,fnp.concatenate((old_coeffs,birth_coeffs),axis=1),blockv.T,0)
                else:
                    factors.append((None,x,y,ac,bc,cc,z,vv,birth_coeffs,vh,0))
            if trace_parts:
                traces=trace_parts[0]
                for part in trace_parts[1:]:traces=traces+part
                radial3c=traces[:,None]*(3/(fulln*(fulln+2)))
            else:radial3c=fnp.zeros((2,1),dtype=fnp.float32)
            fresh_pivots=pivots
            row_ids=(keep[:,None]*fulln+keep).reshape(-1)
            delta_cov=cov-pc[keep,:][:,keep]
            pc=pc+fnp.bincount(row_ids,weights=delta_cov.reshape(-1),minlength=fulln*fulln).reshape((fulln,fulln)).astype(fnp.float32)
            pm=row
            previous=keep
        if self.mean_translation_fraction:
            dt_output=fnp.stack(rows)*(1-4/(4*fulln)+4/(32*fulln*fulln)+20/(128*fulln*fulln*fulln))+(1-self.mean_translation_fraction)*fnp.einsum('c,lcf->lf',channel_weights,fnp.stack(channel_history))
        else:
            dt_output=fnp.stack(rows)*(1-4/(4*fulln)+4/(32*fulln*fulln)+20/(128*fulln*fulln*fulln))+fnp.einsum('c,lcf->lf',channel_weights,fnp.stack(channel_history))
        if self.diagonal_transport_enabled:dt_output=dt_output+(fnp.stack(dt_corrections)/fulln).astype(fnp.float32)
        if _cr24_extra is not None:dt_output=dt_output+fnp.concatenate((fnp.zeros((mlp.depth-1,fulln),dtype=fnp.float32),_cr24_extra.astype(fnp.float32)[None,:]),axis=0)
        return dt_output
