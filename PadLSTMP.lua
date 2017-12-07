local PadLSTMP, parent = torch.class('srnn.PadLSTMP', 'srnn.AbstractLenCell')

function PadLSTMP:buildStep(inputSize, outputSize, layers, dropout)

	return srnn.units.getLSTMPUnit(inputSize, outputSize, layers, dropout)

end
