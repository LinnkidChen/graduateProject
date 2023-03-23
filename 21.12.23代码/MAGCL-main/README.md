Source code for Model Augmented Graph Contrastive Learning

For example, to run MA-GCL under Cora, execute:

    python main.py --device cuda:0 --dataset Cora
    
    python main.py --dataset ACM --device cuda:0 >> acmLog.txt  


start of epoch  1
tensor([[   0,    1,    2,  ..., 6563, 6563, 6563],
        [   0,    1,    2,  ..., 4586, 6497, 6563]], device='cuda:0') torch.Size([2, 21976])
tensor([1., 1., 1.,  ..., 1., 1., 1.], device='cuda:0') torch.Size([21976])

tensor([[   0,    0,    0,  ..., 4017, 4017, 4018],
        [   0,    8,   20,  ..., 3992, 4017, 4018]], device='cuda:0') torch.Size([2, 57853])
tensor([1., 1., 1.,  ..., 1., 1., 1.], device='cuda:0') torch.Size([57853])


tensor([[   0,    0,    0,  ..., 4018, 4018, 4018],
        [   0,   75,  586,  ..., 4015, 4017, 4018]]) torch.Size([2, 4338213]) torch.LongTensor
tensor([1., 1., 1.,  ..., 1., 1., 1.]) torch.Size([4338213]) torch.FloatTensor
1
4019

start of epoch  1
tensor([[   0,    1,    2,  ..., 6563, 6563, 6563],
        [   0,    1,    2,  ..., 4586, 6497, 6563]]) torch.Size([2, 21976]) torch.LongTensor
tensor([1., 1., 1.,  ..., 1., 1., 1.]) torch.Size([21976]) torch.FloatTensor
1
4019