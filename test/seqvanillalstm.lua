require "srnn"

function test(bsize, isize, osize, nlayer, nstep, rs, tsfm)
	local id={}
	local gd={}
	for _ = 1, nstep do
		table.insert(id, torch.randn(bsize, isize))
		table.insert(gd, torch.randn(bsize, osize))
	end
	local tmod_core=srnn.StepVanillaLSTM(isize, osize, nlayer)
	local tmod=srnn.Sequencer(tmod_core, osize, nil, rs, tsfm)
	tmod:evaluate()
	for i = 1, 3 do
		tmod:forward(id)
	end
	tmod:training()
	tmod:forward(id)
	tmod:backward(id, gd)
	tmod:forward(id)
	tmod:updateGradInput(id, gd)
	tmod:accGradParameters(id, gd)
	id=torch.randn(nstep, bsize, isize)
	gd=torch.randn(nstep, bsize, osize)
	tmod_core=srnn.StepVanillaLSTM(isize, osize, nlayer)
	tmod=srnn.Sequencer(tmod_core, osize, nil, rs, tsfm)
	tmod:evaluate()
	for i = 1, 3 do
		tmod:forward(id)
	end
	tmod:training()
	tmod:forward(id)
	tmod:backward(id, gd)
	tmod:forward(id)
	tmod:updateGradInput(id, gd)
	tmod:accGradParameters(id, gd)
end

test(30, 29, 59, 1, 1)
test(59, 26, 29, 3, 7)
test(30, 29, 59, 1, 1, true, true)
test(59, 26, 29, 3, 7, true, true)
