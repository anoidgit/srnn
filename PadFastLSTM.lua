local PadFastLSTM, parent = torch.class('srnn.PadFastLSTM', 'srnn.AbstractLenCell')

function PadFastLSTM:buildStep(inputSize, outputSize, layers, dropout)

	return srnn.units.getFastLSTMUnit(inputSize, outputSize, layers, dropout)

end
