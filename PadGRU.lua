local PadGRU, parent = torch.class('srnn.PadGRU', 'srnn.AbstractLenRecurrent')

function PadGRU:buildStep(inputSize, outputSize, layers, dropout)

	return srnn.units.getGRUUnit(inputSize, outputSize, layers, dropou)

end
