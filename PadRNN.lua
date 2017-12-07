local PadRNN, parent = torch.class('srnn.PadRNN', 'srnn.AbstractLenRecurrent')

function PadRNN:buildStep(inputSize, outputSize, layers, dropout)

	return srnn.units.getRNNUnit(inputSize, outputSize, layers, dropout)

end
