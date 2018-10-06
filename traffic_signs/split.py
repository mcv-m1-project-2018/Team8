import random
from metrics import get_dictionary

def divide_training_test_SL(signal_list):
	signal_sorted = sorted(signal_list, key=lambda x: x.pixels, reverse=True)
	first = 0
	last = 10
	training, test = [], []
	while last<len(signal_sorted):
		part = signal_sorted[first:min(last, len(signal_sorted))]
		random.shuffle(part)
		number_signals = len(part)
		split_position = int(0.7*number_signals)
		training.append(part[:split_position])
		test.append(part[split_position:]) 
		first = first + 10
		last = last + 10

	return training, test


def divide(signals_type_dict):
	for key in signals_type_dict:
		sig_subdict = signals_type_dict[key]
		training, test = divide_training_test_SL(sig_subdict['signal_list'])
		sig_subdict['signal_list'] = {}
		sig_subdict['signal_list']['training'] = training
		sig_subdict['signal_list']['test'] = test                        
	
def main():
	signal_type_dict = get_dictionary()
	for key in signal_type_dict:
		print("Key:", key, "has", len(signal_type_dict[key]['signal_list']))
	print("-----------------")
	
	divide(signal_type_dict)

	for key in signal_type_dict:
		tmp_stl = signal_type_dict[key]['signal_list']
		train = tmp_stl['training']
		test = tmp_stl['test']
		print("Key:", key, "has", len(train)+len(test), "| Training:",len(train), " and Test:",len(test))
	
if __name__ == '__main__':
	main()
    random.shuffle(part)
    number_signals = len(part)
    split_position = int(0.7*number_signals)
    training.append(part[:split_position])
    test.append(part[split_position:]) 
    first = first + 10
    last = last + 10

	return training, test
		