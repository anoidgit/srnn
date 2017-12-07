require "srnn"

function test(bsize, isize, osize, nlayer, nstep, npadstep, rs, tsfm)
	local id={}
	local gd={}
	for _ = 1, nstep do
		table.insert(id, torch.randn(bsize, isize))
		table.insert(gd, torch.randn(bsize, osize))
	end
	local lvec = torch.LongTensor(bsize):fill(npadstep)
	local tmod_core=srnn.PadVanillaLSTM(isize, osize, nlayer)
	local tmod=srnn.PadSequencer(tmod_core, osize, nil, rs, tsfm)
	tmod:evaluate()
	for i = 1, 3 do
		tmod:forward({id, lvec})
	end
	tmod:training()
	tmod:forward({id, lvec})
	tmod:backward({id, lvec}, gd)
	tmod:forward({id, lvec})
	tmod:updateGradInput({id, lvec}, gd)
	tmod:accGradParameters({id, lvec}, gd)
	id=torch.randn(nstep, bsize, isize)
	gd=torch.randn(nstep, bsize, osize)
	tmod_core=srnn.PadVanillaLSTM(isize, osize, nlayer)
	tmod=srnn.PadSequencer(tmod_core, osize, nil, rs, tsfm)
	tmod:evaluate()
	for i = 1, 3 do
		tmod:forward({id, lvec})
	end
	tmod:training()
	tmod:forward({id, lvec})
	tmod:backward({id, lvec}, gd)
	tmod:forward({id, lvec})
	tmod:updateGradInput({id, lvec}, gd)
	tmod:accGradParameters({id, lvec}, gd)
end

test(30, 29, 59, 1, 1, 0)
test(59, 26, 29, 3, 7, 2)
test(30, 29, 59, 1, 1, 0, true, true)
test(59, 26, 29, 3, 7, 3, true, true)
