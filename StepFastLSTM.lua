local StepFastLSTM, parent = torch.class('srnn.StepFastLSTM', 'srnn.AbstractCell')

function StepFastLSTM:buildStep(inputSize, outputSize, layers, dropout)

	return srnn.units.getFastLSTMUnit(inputSize, outputSize, layers, dropout)

end
