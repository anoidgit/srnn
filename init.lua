require('torch')
require('nn')
require('nngraph')

srnn = {} -- define the global srnn table
srnn.units = {} -- prepare for units of RNNs

require('srnn.units.getRNNUnit')
require('srnn.units.getFastLSTMUnit')
require('srnn.units.getLSTMPUnit')
require('srnn.units.getVanillaLSTMUnit')
require('srnn.units.getGRUUnit')

require('srnn.ValueMinus')

require('srnn.SequenceContainer')

require('srnn.RecurrentContainer')

require('srnn.AbstractRecurrent')
require('srnn.AbstractRecurrentCell')

require('srnn.StepRNN')
require('srnn.StepFastLSTM')
require('srnn.StepLSTMP')
require('srnn.StepVanillaLSTM')
require('srnn.StepGRU')

require('srnn.Sequencer')

return srnn
