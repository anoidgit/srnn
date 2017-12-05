local StepRNN, parent = torch.class('srnn.StepRNN', 'srnn.AbstractRecurrent')

function StepRNN:buildStep(inputSize, outputSize, layers, dropout)

	return srnn.units.getRNNUnit(inputSize, outputSize, layers, dropout)

end
