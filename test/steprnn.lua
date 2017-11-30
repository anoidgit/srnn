require "srnn"

function test(bsize, isize, osize, nlayer, nstep)
	local id={}
	local gd={}
	for _ = 1, nstep do
		table.insert(id, torch.randn(bsize, isize))
		table.insert(gd, torch.randn(bsize, osize))
	end
	local tmod=srnn.StepRNN(isize, osize, nlayer)
	tmod:evaluate()
	for _, v in ipairs(id) do
		print("Step:".._)
		print("tmod forward step:"..tmod.fwd_step)
		tmod:forward(v)
	end
	tmod:training()
	for _, v in ipairs(id) do
		print("Step:".._)
		print("tmod forward step:"..tmod.fwd_step)
		tmod:forward(v)
	end
	for _ = #id, 1, -1 do
		tmod:backward(id[_], gd[_])
		print("tmod next backward step:"..tmod.ugi_step)
	end
	for _, v in ipairs(id) do
		print("Step:".._)
		print("tmod forward step:"..tmod.fwd_step)
		tmod:forward(v)
	end
	for _ = #id, 1, -1 do
		tmod:updateGradInput(id[_], gd[_])
		print("tmod next updateGradInput step:"..tmod.ugi_step)
	end
	for _ = #id, 1, -1 do
		tmod:accGradParameters(id[_], gd[_])
		print("tmod next accGradParameters step:"..tmod.acg_step)
	end
end

test(30, 29, 59, 1, 1)
test(59, 26, 29, 3, 7)
