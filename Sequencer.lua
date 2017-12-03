local Sequencer, parent = torch.class('srnn.Sequencer', 'nn.Container')

function Sequencer:__init(module, outputSize, stepClone, reverseSeq, transferMod, resetStepEva)
	parent.__init(self)
	self.outputSize = outputSize
	self.stepClone = ((stepClone == nil) and (not torch.isTypeOf(module, 'srnn.SequenceContainer'))) and true or stepClone
	self.reverseSeq = reverseSeq
	self.transferMod = transferMod
	if self.stepClone and (not torch.isTypeOf(module, 'srnn.SequenceContainer')) then
		self.network = srnn.SequenceContainer(module)
	else
		self.network = module
	end
	self:add(module)
	if self.network.resetStep then
		self.resetStepEva = (resetStepEva == nil) and true or resetStepEva
	end
	self:training()
end

function Sequencer:updateOutput(input)
	self:prepareSelf(input)
	if self.reverseSeq then
		if stepClone then
			local np = self.nElement + 1
			if self.outputTable and (not self.train) then
				for _ = self.nElement, 1, -1 do
					self.output[_] = self.network:net(np - _):updateOutput(input[_]):clone()
				end
			else
				for _ = self.nElement, 1, -1 do
					self.output[_] = self.network:net(np - _):updateOutput(input[_])
				end
			end
		else
			if self.outputTable and (not self.train) then
				for _ = self.nElement, 1, -1 do
					self.output[_] = self.network:updateOutput(input[_]):clone()
				end
			else
				for _ = self.nElement, 1, -1 do
					self.output[_] = self.network:updateOutput(input[_])
				end
			end
		end
	else
		if stepClone then
			if self.outputTable and (not self.train) then
				for _ = 1, self.nElement do
					self.output[_] = self.network:net(_):updateOutput(input[_]):clone()
				end
			else
				for _ = 1, self.nElement do
					self.output[_] = self.network:net(_):updateOutput(input[_])
				end
			end
		else
			if self.outputTable and (not self.train) then
				for _ = 1, self.nElement do
					self.output[_] = self.network:updateOutput(input[_]):clone()
				end
			else
				for _ = 1, self.nElement do
					self.output[_] = self.network:updateOutput(input[_])
				end
			end
		end
	end
	return self.output
end

function Sequencer:updateGradInput(input, gradOutput)
	if self.reverseSeq then
		if stepClone then
			local np = self.nElement + 1
			for _ = self.nElement, 1, -1 do
				self.gradInput[_] = self.network:net(np - _):updateGradInput(input[_], gradOutput[_])
			end
		else
			for _ = self.nElement, 1, -1 do
				self.gradInput[_] = self.network:updateGradInput(input[_], gradOutput[_])
			end
		end
	else
		if stepClone then
			for _ = 1, self.nElement do
				self.gradInput[_] = self.network:net(_):updateGradInput(input[_], gradOutput[_])
			end
		else
			for _ = 1, self.nElement do
				self.gradInput[_] = self.network:updateGradInput(input[_], gradOutput[_])
			end
		end
	end
	return self.gradInput
end

function Sequencer:accGradParameters(input, gradOutput, scale)
	if self.reverseSeq then
		if stepClone then
			local np = self.nElement + 1
			for _ = self.nElement, 1, -1 do
				self.network:net(np - _):accGradParameters(input[_], gradOutput[_], scale)
			end
		else
			for _ = self.nElement, 1, -1 do
				self.network:accGradParameters(input[_], gradOutput[_], scale)
			end
		end
	else
		if stepClone then
			for _ = 1, self.nElement do
				self.network:net(_):accGradParameters(input[_], gradOutput[_], scale)
			end
		else
			for _ = 1, self.nElement do
				self.network:accGradParameters(input[_], gradOutput[_], scale)
			end
		end
	end
end

function Sequencer:backward(input, gradOutput, scale)
	if self.reverseSeq then
		if stepClone then
			local np = self.nElement + 1
			for _ = self.nElement, 1, -1 do
				self.gradInput[_] = self.network:net(np - _):backward(input[_], gradOutput[_], scale)
			end
		else
			for _ = self.nElement, 1, -1 do
				self.gradInput[_] = self.network:backward(input[_], gradOutput[_], scale)
			end
		end
	else
		if stepClone then
			for _ = 1, self.nElement do
				self.gradInput[_] = self.network:net(_):backward(input[_], gradOutput[_], scale)
			end
		else
			for _ = 1, self.nElement do
				self.gradInput[_] = self.network:backward(input[_], gradOutput[_], scale)
			end
		end
	end
	return self.gradInput
end

function Sequencer:prepareSelf(input)
	if torch.isTensor(input) then
		self.nElement = input:size(1)
		if self.transferMod then
			self.output = {}
			self.outputTable = true
		else
			self.output:resize(input:size(1), input:size(2), self.outputSize)
			self.outputTable = false
		end
		if self.train then
			self.gradInput:resizeAs(input)
		end
	else
		self.nElement = #input
		if self.transferMod then
			self.output:resize(self.nElement, input[1]:size(1), self.outputSize)
			self.outputTable = false
		else
			self.output = {}
			self.outputTable = true
		end
		if self.train then
			self.gradInput = {}
		end
	end
	if self.resetStepEva and (not self.train) then
		self.network:resetStep()
	end
end

function Sequencer:clearState()
	self.nElement = nil
	self.outputTable = nil
	return parent.clearState(self)
end
