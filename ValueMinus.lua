local ValueMinus, parent = torch.class('srnn.ValueMinus', 'nn.Module')

function ValueMinus:__init(value)
	parent.__init(self)
	self.value = value
end

function ValueMinus:updateOutput(input)
	self.output:resizeAs(input):fill(self.value):csub(input)
	return self.output
end

function ValueMinus:updateGradInput(input, gradOutput)
	self.gradInput:resizeAs(gradOutput):neg(gradOutput)
	return self.gradInput
end
