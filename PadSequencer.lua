local PadSequencer, parent = torch.class('srnn.PadSequencer', 'nn.Container')

function PadSequencer:__init(module, outputSize, stepClone, reverseSeq, transferMod, resetStepEva)
	parent.__init(self)
	self.gradInput = {}
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

function PadSequencer:updateOutput(input)
	-- input should be a table, the first element is the real input and the second element is the length vector to describe how many data have been padded in the head of the sequence
	local _input, _len = unpack(input)
	self:prepareSelf(_input)
	if self.reverseSeq then
		if stepClone then
			local np = self.nElement + 1
			if self.outputTable and (not self.train) then
				for _ = self.nElement, 1, -1 do
					self.output[_] = self.network:net(np - _):updateOutput({_input[_], _len}):clone()
				end
			else
				for _ = self.nElement, 1, -1 do
					self.output[_] = self.network:net(np - _):updateOutput({_input[_], _len})
				end
			end
		else
			if self.outputTable and (not self.train) then
				for _ = self.nElement, 1, -1 do
					self.output[_] = self.network:updateOutput({_input[_], _len}):clone()
				end
			else
				for _ = self.nElement, 1, -1 do
					self.output[_] = self.network:updateOutput({_input[_], _len})
				end
			end
		end
	else
		if stepClone then
			if self.outputTable and (not self.train) then
				for _ = 1, self.nElement do
					self.output[_] = self.network:net(_):updateOutput({_input[_], _len}):clone()
				end
			else
				for _ = 1, self.nElement do
					self.output[_] = self.network:net(_):updateOutput({_input[_], _len})
				end
			end
		else
			if self.outputTable and (not self.train) then
				for _ = 1, self.nElement do
					self.output[_] = self.network:updateOutput({_input[_], _len}):clone()
				end
			else
				for _ = 1, self.nElement do
					self.output[_] = self.network:updateOutput({_input[_], _len})
				end
			end
		end
	end
	return self.output
end

function PadSequencer:updateGradInput(input, gradOutput)
	local _input, _len = unpack(input)
	if self.reverseSeq then
		if stepClone then
			local np = self.nElement + 1
			for _ = self.nElement, 1, -1 do
				self.gradInput[1][_] = self.network:net(np - _):updateGradInput({_input[_], _len}, gradOutput[_])[1]
			end
		else
			for _ = self.nElement, 1, -1 do
				self.gradInput[1][_] = self.network:updateGradInput({_input[_], _len}, gradOutput[_])[1]
			end
		end
	else
		if stepClone then
			for _ = 1, self.nElement do
				self.gradInput[1][_] = self.network:net(_):updateGradInput({_input[_], _len}, gradOutput[_])[1]
			end
		else
			for _ = 1, self.nElement do
				self.gradInput[1][_] = self.network:updateGradInput({_input[_], _len}, gradOutput[_])[1]
			end
		end
	end
	if not self.gradInput[2] then
		self.gradInput[2] = input[2].new()
	end
	self.gradInput[2]:resizeAs(input[2]):zero()
	return self.gradInput
end

function PadSequencer:accGradParameters(input, gradOutput, scale)
	local _input, _len = unpack(input)
	if self.reverseSeq then
		if stepClone then
			local np = self.nElement + 1
			for _ = self.nElement, 1, -1 do
				self.network:net(np - _):accGradParameters({_input[_], _len}, gradOutput[_], scale)
			end
		else
			for _ = self.nElement, 1, -1 do
				self.network:accGradParameters({_input[_], _len}, gradOutput[_], scale)
			end
		end
	else
		if stepClone then
			for _ = 1, self.nElement do
				self.network:net(_):accGradParameters({_input[_], _len}, gradOutput[_], scale)
			end
		else
			for _ = 1, self.nElement do
				self.network:accGradParameters({_input[_], _len}, gradOutput[_], scale)
			end
		end
	end
end

function PadSequencer:backward(input, gradOutput, scale)
	local _input, _len = unpack(input)
	if self.reverseSeq then
		if stepClone then
			local np = self.nElement + 1
			for _ = self.nElement, 1, -1 do
				self.gradInput[1][_] = self.network:net(np - _):backward({_input[_], _len}, gradOutput[_], scale)[1]
			end
		else
			for _ = self.nElement, 1, -1 do
				self.gradInput[1][_] = self.network:backward({_input[_], _len}, gradOutput[_], scale)[1]
			end
		end
	else
		if stepClone then
			for _ = 1, self.nElement do
				self.gradInput[1][_] = self.network:net(_):backward({_input[_], _len}, gradOutput[_], scale)[1]
			end
		else
			for _ = 1, self.nElement do
				self.gradInput[1][_] = self.network:backward({_input[_], _len}, gradOutput[_], scale)[1]
			end
		end
	end
	if not self.gradInput[2] then
		self.gradInput[2] = input[2].new()
	end
	self.gradInput[2]:resizeAs(input[2]):zero()
	return self.gradInput
end

function PadSequencer:prepareSelf(input)
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
			if not self.gradInput[1] then
				self.gradInput[1] = input.new()
			end
			self.gradInput[1]:resizeAs(input)
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
			self.gradInput[1] = {}
		end
	end
	if self.resetStepEva and (not self.train) then
		self.network:resetStep()
	end
end

function PadSequencer:clearState()
	self.gradInput = {}
	self.nElement = nil
	self.outputTable = nil
	return parent.clearState(self)
end
