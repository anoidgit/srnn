local StepGRU, parent = torch.class('srnn.StepGRU', 'srnn.AbstractRecurrent')

function StepGRU:buildStep(inputSize, outputSize, layers, dropout)

	return srnn.units.getGRUUnit(inputSize, outputSize, layers, dropout)

end
