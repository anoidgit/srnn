require('torch')
require('nn')
require('nngraph')

srnn = {} -- define the global srnn table

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

return srnn
