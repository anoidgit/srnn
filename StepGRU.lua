local StepGRU, parent = torch.class('srnn.StepGRU', 'srnn.AbstractRecurrent')

--[[
z[t] = σ(W[x->z]x[t] + W[s->z]s[t−1] + b[1->z])            (1)
r[t] = σ(W[x->r]x[t] + W[s->r]s[t−1] + b[1->r])            (2)
h[t] = tanh(W[x->h]x[t] + W[hr->c](s[t−1]r[t]) + b[1->h])  (3)
s[t] = (1-z[t])h[t] + z[t]s[t-1]                           (4)
]]
function StepGRU:buildStep(inputSize, outputSize, layers, dropout)
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

		-- calc input gate and forget gate
		local if_gates = nn.Sigmoid()(nn.Linear(isize + outputSize, outputSize * 2)(io_concat))
		local i_gate = nn.Narrow(2, 1, outputSize)(if_gates)
		local f_gate = nn.Narrow(2, outputSize + 1, outputSize)(if_gates)

		local hidden_input = nn.JoinTable(2, 2)({input, nn.CMulTable()({f_gate, prev_output})})
		-- calc hidden state with input and previous output
		local hidden = nn.Tanh()(nn.Linear(isize + outputSize, outputSize)(hidden_input))

		-- generate a new output
		local next_out = nn.CAddTable()({nn.CMulTable()({srnn.ValueMinus(1)(i_gate), hidden}), nn.CMulTable()({i_gate, prev_output})})
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
