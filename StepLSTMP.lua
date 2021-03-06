local StepLSTMP, parent = torch.class('srnn.StepLSTMP', 'srnn.AbstractCell')

function StepLSTMP:buildStep(inputSize, outputSize, layers, dropout)

	return srnn.units.getLSTMPUnit(inputSize, outputSize, layers, dropout)

end
