local StepVanillaLSTM, parent = torch.class('srnn.StepVanillaLSTM', 'srnn.AbstractRecurrentCell')

function StepVanillaLSTM:buildStep(inputSize, outputSize, layers, dropout)

	return srnn.units.getVanillaLSTMUnit(inputSize, outputSize, layers, dropout)

end
