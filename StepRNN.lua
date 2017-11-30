local StepRNN, parent = torch.class('srnn.StepRNN', 'srnn.AbstractRecurrent')

--[[
h[t] = tanh(W[x->h]x[t] + W[hr->h]h[t-1] + b[1->h])  (1)
]]
function StepRNN:buildStep(inputSize, outputSize, layers, dropout)
	local input -- the input of a step
	local inputs = {} -- real input to this step
	local init_outs = {} -- previous output for each layer
	local outputs = {} -- outputs generated for each layer

	-- start build step module
	local isize = inputSize -- for the first layer, the dimension of input is inputSize, otherwise it is outputSize

	for i = 1, layers do
		-- set up input for layer i
		if i == 1 then
			if dropout and dropout > 0 and dropout < 1 then
				input = nn.Dropout(dropout)()
			else
				input = nn.Identity()()
			end
			table.insert(inputs, input)
		elseif dropout and dropout > 0 and dropout < 1 then
				input = nn.Dropout(dropout)(input)
		end

		-- set up previous output and cell for layer i
		local prev_output = nn.Identity()()
		table.insert(init_outs, prev_output)

		-- concat input and previous output
		local io_concat = nn.JoinTable(2, 2)({input, prev_output})

		-- generate a new output
		local next_out = nn.Tanh()(nn.Linear(isize + outputSize, outputSize)(io_concat))
		table.insert(outputs, next_out)

		-- update input for the next layer
		input = next_out
		isize = outputSize
	end

	-- update inputs and outputs to generate a step module
	-- order: (input, )(init_)outputs and (init_)cells
	for _, unit in ipairs(init_outs) do
		table.insert(inputs, unit)
	end
	-- generate a step module
	return nn.gModule(inputs, outputs)

end
