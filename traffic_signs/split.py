import random
from numpy import mean
from metrics import get_dictionary

def divide_training_validation_SL(signal_list):
	random.seed(1)
	signal_sorted = sorted(signal_list, key=lambda x: x.pixels, reverse=True)
	first = 0
	last = 10
	training, validation = [], []
	while last<len(signal_sorted)+10:
		part = signal_sorted[first:min(last, len(signal_sorted))]
		random.shuffle(part)
		number_signals = len(part)
		split_position = round(0.7*number_signals)
		training.extend(part[:split_position])
		validation.extend(part[split_position:]) 
		first = first + 10
		last = last + 10

	return training, validation


def divide(signals_type_dict):
	for key in signals_type_dict:
		sig_subdict = signals_type_dict[key]
		training, validation = divide_training_validation_SL(sig_subdict['signal_list'])
		sig_subdict['signal_list'] = {}
		sig_subdict['signal_list']['training'] = training
		sig_subdict['signal_list']['validation'] = validation
	
	
def main():
	signal_type_dict = get_dictionary()
	
	for key in signal_type_dict:
		print("Key:", key, "has", len(signal_type_dict[key]['signal_list']))
	print("-----------------")
	
	divide(signal_type_dict)

	for key in signal_type_dict:
		tmp_stl = signal_type_dict[key]['signal_list']
		train = len(tmp_stl['training'])
		validation = len(tmp_stl['validation'])
		total = train+validation

		trainPixelsMean = mean([signal.pixels for signal in tmp_stl['training']])
		validPixelsMean = mean([signal.pixels for signal in tmp_stl['validation']])

		print("Key:", key, "has", total, "| Training ("+"{0:.2f}".format((train/total)*100)+"%):", train, \
							  		 " and validation("+"{0:.2f}".format((validation/total)*100)+"%):",validation, end="")
		
		print("   \t|","{0:.0f}".format(trainPixelsMean)," - ", "{0:.0f}".format(validPixelsMean))
	
if __name__ == '__main__':
	main()
		