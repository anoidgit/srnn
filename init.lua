require('torch')
require('nn')
require('nngraph')

srnn = {} -- define the global srnn table
srnn.units = {} -- prepare for units of RNNs

require('srnn.ValueMinus')

require('srnn.units.getRNNUnit')
require('srnn.units.getFastLSTMUnit')
require('srnn.units.getLSTMPUnit')
require('srnn.units.getVanillaLSTMUnit')
require('srnn.units.getGRUUnit')

require('srnn.SequenceContainer')

require('srnn.RecurrentContainer')

require('srnn.LenRecurrentContainer')

require('srnn.AbstractRecurrent')
require('srnn.AbstractCell')

require('srnn.AbstractLenRecurrent')
require('srnn.AbstractLenCell')

require('srnn.StepRNN')
require('srnn.StepFastLSTM')
require('srnn.StepLSTMP')
require('srnn.StepVanillaLSTM')
require('srnn.StepGRU')

require('srnn.PadRNN')
require('srnn.PadFastLSTM')
require('srnn.PadLSTMP')
require('srnn.PadVanillaLSTM')
require('srnn.PadGRU')

require('srnn.Sequencer')

require('srnn.PadSequencer')

return srnn
