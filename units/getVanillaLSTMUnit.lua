--[[
i[t] = σ(W[x->i]x[t] + W[h->i]h[t−1] + W[c->i]c[t−1] + b[1->i])      (1)
f[t] = σ(W[x->f]x[t] + W[h->f]h[t−1] + W[c->f]c[t−1] + b[1->f])      (2)
z[t] = tanh(W[x->c]x[t] + W[h->c]h[t−1] + b[1->c])                   (3)
c[t] = f[t]c[t−1] + i[t]z[t]                                         (4)
o[t] = σ(W[x->o]x[t] + W[h->o]h[t−1] + W[c->o]c[t] + b[1->o])        (5)
h[t] = o[t]tanh(c[t])                                                (6)
]]
function srnn.units.getVanillaLSTMUnit(inputSize, outputSize, layers, dropout)
	local input -- the input of a step
	local inputs = {} -- real input to this step
	local init_cells = {} -- previous cells for each layer
	local init_outs = {} -- previous output for each layer
	local outputs = {} -- outputs generated for each layer
	local cells = {} -- cells generated for each layer

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
		local prev_cell = nn.Identity()()
		table.insert(init_cells, prev_cell)

		-- concat input and previous output
		local io_concat = nn.JoinTable(2, 2)({input, prev_output})

		-- concat input with previous output and cell
		local all_concat = nn.JoinTable(2, 2)({io_concat, prev_cell})

		-- calc input gate and forget gate
		local if_gates = nn.Sigmoid()(nn.Linear(isize + outputSize * 2, outputSize * 2)(all_concat))
		local i_gate = nn.Narrow(2, 1, outputSize)(if_gates)
		local f_gate = nn.Narrow(2, outputSize + 1, outputSize)(if_gates)

		-- calc hidden state with input and previous output
		local hidden = nn.Tanh()(nn.Linear(isize + outputSize, outputSize)(io_concat))

		-- update cell with previous cell, hidden state, input gate and forget gate
		local next_cell = nn.CAddTable()({nn.CMulTable()({f_gate, prev_cell}), nn.CMulTable()({i_gate, hidden})})
		table.insert(cells, next_cell)

		-- concat input with previous output and new cell
		local all_concat_new = nn.JoinTable(2, 2)({io_concat, next_cell})

		-- calc output gate with new data concated
		local o_gate = nn.Sigmoid()(nn.Linear(isize + outputSize * 2, outputSize)(all_concat_new))
		-- generate a new output
		local next_out = nn.CMulTable()({o_gate, nn.Tanh()(next_cell)})
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
	for _, unit in ipairs(init_cells) do
		table.insert(inputs, unit)
	end
	for _, unit in ipairs(cells) do
		table.insert(outputs, unit)
	end

	-- generate a step module
	return nn.gModule(inputs, outputs)

end
