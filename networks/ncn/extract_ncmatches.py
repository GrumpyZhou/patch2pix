import torch
import torch.nn
from torch.autograd import Variable
import numpy as np

def corr_to_matches(corr4d, delta4d=None, ksize=1, do_softmax=True, scale='positive', 
                    invert_matching_direction=False, return_indices=True):
    to_cuda = lambda x: x.to(corr4d.device) if corr4d.is_cuda else x        
    batch_size,ch,fs1,fs2,fs3,fs4 = corr4d.size()  # b, c, h, w, h, w
    if scale=='centered':
        XA,YA=np.meshgrid(np.linspace(-1,1,fs2*ksize),np.linspace(-1,1,fs1*ksize))
        XB,YB=np.meshgrid(np.linspace(-1,1,fs4*ksize),np.linspace(-1,1,fs3*ksize))
    elif scale=='positive':
        # Upsampled resolution linear space
        XA,YA=np.meshgrid(np.linspace(0,1,fs2*ksize),np.linspace(0,1,fs1*ksize))
        XB,YB=np.meshgrid(np.linspace(0,1,fs4*ksize),np.linspace(0,1,fs3*ksize))
    # Index meshgrid for current resolution
    JA,IA=np.meshgrid(range(fs2),range(fs1)) 
    JB,IB=np.meshgrid(range(fs4),range(fs3))
    
    XA,YA=Variable(to_cuda(torch.FloatTensor(XA))),Variable(to_cuda(torch.FloatTensor(YA)))
    XB,YB=Variable(to_cuda(torch.FloatTensor(XB))),Variable(to_cuda(torch.FloatTensor(YB)))

    JA,IA=Variable(to_cuda(torch.LongTensor(JA).view(1,-1))),Variable(to_cuda(torch.LongTensor(IA).view(1,-1)))
    JB,IB=Variable(to_cuda(torch.LongTensor(JB).view(1,-1))),Variable(to_cuda(torch.LongTensor(IB).view(1,-1)))
    
    if invert_matching_direction:
        nc_A_Bvec=corr4d.view(batch_size,fs1,fs2,fs3*fs4)

        if do_softmax:
            nc_A_Bvec=torch.nn.functional.softmax(nc_A_Bvec,dim=3)

        # Max and argmax
        match_A_vals,idx_A_Bvec=torch.max(nc_A_Bvec,dim=3)
        score=match_A_vals.view(batch_size,-1)
        
        # Pick the indices for the best score
        iB=IB.view(-1)[idx_A_Bvec.view(-1)].view(batch_size,-1).contiguous()  # b, h1*w1
        jB=JB.view(-1)[idx_A_Bvec.view(-1)].view(batch_size,-1).contiguous()
        iA=IA.expand_as(iB).contiguous()
        jA=JA.expand_as(jB).contiguous()
        
    else:    
        nc_B_Avec=corr4d.view(batch_size,fs1*fs2,fs3,fs4) # [batch_idx,k_A,i_B,j_B]
        if do_softmax:
            nc_B_Avec=torch.nn.functional.softmax(nc_B_Avec,dim=1)

        match_B_vals,idx_B_Avec=torch.max(nc_B_Avec,dim=1)
        score=match_B_vals.view(batch_size,-1)
        
        iA=IA.view(-1)[idx_B_Avec.view(-1)].view(batch_size,-1).contiguous() # b, h2*w2
        jA=JA.view(-1)[idx_B_Avec.view(-1)].view(batch_size,-1).contiguous() 
        iB=IB.expand_as(iA).contiguous()
        jB=JB.expand_as(jA).contiguous()
    
    if delta4d is not None: # relocalization, it is also the case ksize > 1
        # The shift within the pooling window reference to (0,0,0,0)
        delta_iA, delta_jA, delta_iB, delta_jB = delta4d  # b, 1, h1, w1, h2, w2 
        
        """ Original implementation
        # Reorder the indices according 
        diA = delta_iA.squeeze(0).squeeze(0)[iA.view(-1), jA.view(-1), iB.view(-1), jB.view(-1)] 
        djA = delta_jA.squeeze(0).squeeze(0)[iA.view(-1), jA.view(-1), iB.view(-1), jB.view(-1)]        
        diB = delta_iB.squeeze(0).squeeze(0)[iA.view(-1), jA.view(-1), iB.view(-1), jB.view(-1)]
        djB = delta_jB.squeeze(0).squeeze(0)[iA.view(-1), jA.view(-1), iB.view(-1), jB.view(-1)]

        # *ksize place the pixel to the 1st location in upsampled 4D-Volumn
        iA = iA * ksize + diA.expand_as(iA)
        jA = jA * ksize + djA.expand_as(jA)
        iB = iB * ksize + diB.expand_as(iB)
        jB = jB * ksize + djB.expand_as(jB)
        """
        
        # Support batches
        for ibx in range(batch_size):
            diA = delta_iA[ibx][0][iA[ibx], jA[ibx], iB[ibx], jB[ibx]]  # h*w
            djA = delta_jA[ibx][0][iA[ibx], jA[ibx], iB[ibx], jB[ibx]]
            diB = delta_iB[ibx][0][iA[ibx], jA[ibx], iB[ibx], jB[ibx]]
            djB = delta_jB[ibx][0][iA[ibx], jA[ibx], iB[ibx], jB[ibx]]
            
            iA[ibx] = iA[ibx] * ksize + diA
            jA[ibx] = jA[ibx] * ksize + djA
            iB[ibx] = iB[ibx] * ksize + diB
            jB[ibx] = jB[ibx] * ksize + djB

    xA = XA[iA.view(-1), jA.view(-1)].view(batch_size, -1)
    yA = YA[iA.view(-1), jA.view(-1)].view(batch_size, -1)
    xB = XB[iB.view(-1), jB.view(-1)].view(batch_size, -1)
    yB = YB[iB.view(-1), jB.view(-1)].view(batch_size, -1)
        
    if return_indices:
        return (jA,iA,jB,iB,score)
    else:
        return (xA,yA,xB,yB,score)    
    
def corr_to_matches_topk(corr4d, delta4d=None, topk=1, ksize=1, do_softmax=True,                     
                         invert_matching_direction=False):

    device = corr4d.device
    batch_size, ch, fs1, fs2, fs3, fs4 = corr4d.size()  # b, c, h, w, h, w

    # Index meshgrid for current resolution
    JA, IA = np.meshgrid(range(fs2), range(fs1)) 
    JB, IB = np.meshgrid(range(fs4), range(fs3))    
    JA, IA = torch.LongTensor(JA).view(1,-1).to(device), torch.LongTensor(IA).view(1,-1).to(device)
    JB, IB = torch.LongTensor(JB).view(1,-1).to(device), torch.LongTensor(IB).view(1,-1).to(device)

    if invert_matching_direction:
        nc_A_Bvec = corr4d.view(batch_size, fs1, fs2, fs3 * fs4)

        if do_softmax:
            nc_A_Bvec = torch.nn.functional.softmax(nc_A_Bvec, dim=3)

        # Max and argmax
        match_A_vals, idx_A_Bvec = torch.topk(nc_A_Bvec, topk, dim=3, largest=True, sorted=True)    
        score = match_A_vals.view(batch_size, -1)

        # Pick the indices for the best score
        iB = IB.view(-1)[idx_A_Bvec.view(-1)].view(batch_size, -1, topk).contiguous()
        jB = JB.view(-1)[idx_A_Bvec.view(-1)].view(batch_size, -1, topk).contiguous()
        iA = IA.unsqueeze(-1).expand_as(iB).contiguous()
        jA = JA.unsqueeze(-1).expand_as(jB).contiguous()

    else:    
        nc_B_Avec = corr4d.view(batch_size, fs1 * fs2, fs3, fs4) # [batch_idx,k_A,i_B,j_B]
        if do_softmax:
            nc_B_Avec = torch.nn.functional.softmax(nc_B_Avec, dim=1)

        match_B_vals, idx_B_Avec = torch.topk(nc_B_Avec, topk, dim=1, largest=True, sorted=True)
        score = match_B_vals.view(batch_size, -1)

        iA = IA.view(-1)[idx_B_Avec.view(-1)].view(batch_size, topk, -1).contiguous()
        jA = JA.view(-1)[idx_B_Avec.view(-1)].view(batch_size, topk, -1).contiguous() 
        iB = IB.unsqueeze(1).expand_as(iA).contiguous() 
        jB = JB.unsqueeze(1).expand_as(jA).contiguous()
        
    iA = iA.view(batch_size, -1)
    jA = jA.view(batch_size, -1)
    iB = iB.view(batch_size, -1)
    jB = jB.view(batch_size, -1)   

    if delta4d is not None: # relocalization, it is also the case ksize > 1
        # The shift within the pooling window reference to (0,0,0,0)
        delta_iA, delta_jA, delta_iB, delta_jB = delta4d

        # Support batches
        for ibx in range(batch_size):
            diA = delta_iA[ibx][0][iA[ibx], jA[ibx], iB[ibx], jB[ibx]]  # h*w
            djA = delta_jA[ibx][0][iA[ibx], jA[ibx], iB[ibx], jB[ibx]]
            diB = delta_iB[ibx][0][iA[ibx], jA[ibx], iB[ibx], jB[ibx]]
            djB = delta_jB[ibx][0][iA[ibx], jA[ibx], iB[ibx], jB[ibx]]
            
            iA[ibx] = iA[ibx] * ksize + diA
            jA[ibx] = jA[ibx] * ksize + djA
            iB[ibx] = iB[ibx] * ksize + diB
            jB[ibx] = jB[ibx] * ksize + djB

    return (jA, iA, jB, iB, score)
