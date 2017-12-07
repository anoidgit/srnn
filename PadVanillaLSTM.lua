local PadVanillaLSTM, parent = torch.class('srnn.PadVanillaLSTM', 'srnn.AbstractLenCell')

function PadVanillaLSTM:buildStep(inputSize, outputSize, layers, dropout)

	return srnn.units.getVanillaLSTMUnit(inputSize, outputSize, layers, dropout)

end
