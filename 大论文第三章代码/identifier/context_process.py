import codecs
import pdb
import numpy as np
import nltk
import os
import re
import random
import json
from collections import defaultdict
# from nltk.corpus import stopwords
import argparse

# parser = argparse.ArgumentParser('identifier.py')
# parser.add_argument('--retrieval_file', help='The retrieved contexts from the knowledge base.')
# parser.add_argument('--conll_folder', help='The data folder you want to generate contexts, the code will read train, dev, test data in the folder in conll formatting.')
# parser.add_argument('--lang', help='The language code of the data, for example "en". We have specical processing for Chinese ("zh") and Code-mixed ("mix").')
# parser.add_argument('--use_sentence', action='store_true', help='use  matched sentence in the retrieval results as the contexts')
# parser.add_argument('--use_paragraph_entity', action='store_true', help='use matched sentence and the wiki anchor in the retrieval results as the contexts')
# args = parser.parse_args()


# en_stopwords = stopwords.words('english')
en_stopwords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}


def replace_ZH(text):
    match_regex = re.compile(u'[\u4e00-\u9fa5。，！：《》、（）]{1} +(?<![a-zA-Z])')
    should_replace_list = match_regex.findall(text)
    order_replace_list = sorted(should_replace_list,key=lambda i:len(i),reverse=True)
    for i in order_replace_list:
        if i == u' ':
            continue
        new_i = i.strip()
        text = text.replace(i,new_i)
    return text


def count_file(file,target='train',is_file=False,max_len=100, comment_label = None):
	if not is_file:
		filelist=os.listdir(file)
		for target_file in filelist:
			if 'swp' in target_file:
				continue
			if target in target_file:
				break
		to_write=os.path.join(file,target_file)
	else:
		target_file=file
		to_write=file
	reader=open(to_write,'r',errors = 'ignore')
	lines=reader.readlines()
	sentences=[]
	sentence=[]
	sent_length=[]
	for line in lines:
		line = line.strip()
		if line == '.' or line == ',':
			continue
		if comment_label is not None and line.startswith(comment_label):
			continue
		if line:
			sentence.append(line)
			sentences.append(sentence.copy())
			sentence=[]
	reader.close()
	return to_write,sentences



def write_file(to_write,sentences,max_len=100):
	write_file=to_write
	writer=open(write_file,'w')
	remove_count=0
	for sentence in sentences:
		if len(sentence)>max_len:
			remove_count+=1
			continue
		for word in sentence:
			writer.write(word+'\n')
		writer.write('\n')
	writer.close()
	print(f"Removed {remove_count} sentences that is longer than {max_len}")


def gen_chinese_search_query(sentences):
	res_sents = []
	for sentence in sentences:
		subtoken_length_count = 0
		temp_keyword = ''
		before_word_is_zh = False
		temp_keyword = sentence[0].split()[0]
		before_word_is_zh = len(re.findall(r'[\u4e00-\u9fff]+', sentence[0].split()[0]))>0
		is_eng = len(re.findall(r'[A-Za-z0-9]+',sentence[0].split()[0]))>0
		add_flag=False
		for word in sentence[1:]:
			word = word.split()[0]
			res=re.findall(r'[\u4e00-\u9fff]+', word)
			if not add_flag:
				if is_eng and len(re.findall(r'[A-Za-z0-9]+',word))>0:
					add_flag = True
				is_eng = len(re.findall(r'[A-Za-z0-9]+',word))
			if len(res)>0 and before_word_is_zh:
				before_word_is_zh = True
				temp_keyword+=word
			elif len(res)>0 and not before_word_is_zh:
				before_word_is_zh = True
				temp_keyword+=' '+word
			elif len(res)==0:
				before_word_is_zh = False
				temp_keyword+=' '+word
			else:
				pdb.set_trace()
		res_sents.append(temp_keyword)
	return res_sents


def spliteKeyWord(str):
	regex = r"[\u4e00-\ufaff]|[0-9]+|[^\u4e00-\ufaff]+\'*[a-z]*"
	matches = re.findall(regex, str, re.UNICODE)
	return matches

def replace_zh_space(text):
	match_regex = re.compile(u'[\u4e00-\u9fa5。\.,，:：《》、\(\)（）]{1} +(?<![a-zA-Z])|\d+ +| +\d+|[a-z A-Z]+')
	should_replace_list = match_regex.findall(text)
	order_replace_list = sorted(should_replace_list,key=lambda i:len(i),reverse=True)
	for i in order_replace_list:
		if i == u' ':
			continue
		new_i = i.strip()
		text = text.replace(i,new_i)
	return text

def match_origin_paragraph(sentence, paragraph):
	expression = r'<e:[^>]*>|</e>'
	p = re.compile(expression)
	try:
		removed_paragraph = re.sub(expression, '', paragraph)
		sent_start_pos = removed_paragraph.index(sentence)
	except:
		return sentence
	sentence_length = len(sentence)
	all_special_spans = []
	try:
		for m in p.finditer(paragraph):
			start, end, text = m.start(),m.end(), m.group()
			span_len = end-start
			if end < span_len + sent_start_pos:
				sent_start_pos += span_len
			elif end >= span_len + sent_start_pos and end < span_len + sent_start_pos + sentence_length:sentence
				sentence_length += span_len
			else:
				break
	except:
		return sentence
	final_span_start = sent_start_pos
	final_span_end = sent_start_pos + sentence_length
	parsed_sent = paragraph[final_span_start:final_span_end]
	try:
		assert sentence == re.sub(expression, '', parsed_sent)
	except:
		pdb.set_trace()
	if parsed_sent == None:
		pdb.set_trace()
	return parsed_sent



def gen_sentence(sentence, lang = None):
	if lang is not None and lang == 'zh':
		temp_keyword = ''
		before_word_is_zh = False
		temp_keyword = sentence[0].split()[0]
		before_word_is_zh = len(re.findall(r'[\u4e00-\u9fff]+', sentence[0].split()[0]))>0
		for word in sentence[1:]:
			word = word.split()[0]
			res=re.findall(r'[\u4e00-\u9fff]+', word)
			if len(res)>0 and before_word_is_zh:
				before_word_is_zh = True
				temp_keyword+=word
			elif len(res)>0 and not before_word_is_zh:
				before_word_is_zh = True
				temp_keyword+=' '+word
			elif len(res)==0:
				before_word_is_zh = False
				temp_keyword+=' '+word
			else:
				pdb.set_trace()
		keyword = temp_keyword
	else:
		keyword=' '.join([word.split()[0] for word in sentence])
	return keyword

def process_google(use_xlmr_tokenization,sentences,google_dict,failed_dict, is_conll = False, clean_file = False, full_doc = False, add_eos = False, length_limit = 300, for_luke = False, max_rank = 999, min_rank = 0, lang = None, is_image_retrieval = False, image_ids = None, is_windowed_entity = False, window_size = 0, is_wiki_retrieval = False, space_zh = False, is_test = False, tokenizer=''):
	new_sentences=[]
	max_idx=999
	if is_conll:
		eos = '<EOS> B-X B-X B-X'
	else:
		eos = '<EOS> B-X'
	if for_luke:
		eos = '<EOS> O'
		add_eos = False
	num_context = []
	old_lang = None
	num_context_for_doc = []
	removed_context = 0
	temp_list = []

	for sent_id, sentence in enumerate(sentences):

			# keyword=' '.join([word.split()[0] for word in sentence])划分出keyword:'沙巴航空服务中心'
		#这里需要修改
		keyword = ''.join(sentence)
		if use_xlmr_tokenization:
			subtoken_length_count = 0
			subtoken_length_count += len(tokenizer.tokenize(keyword))
		# if is_wiki_retrieval:
		# 	keyword = keyword.lower()
		# ====================
		# if google_dict[keyword]:
		# 	continue
		if keyword not in google_dict:
			continue
		context = google_dict[keyword].copy()
		context = context[min_rank:max_rank]
		start_word=[]
		end_word=[]
		if len(context)>max_idx*2:
			num_context.append(max_idx*2)
		else:
			num_context.append(len(context))
		current_context_count = 0
		used_context = []
		used_context_length = []
		for cxt_id, cxt in enumerate(context):
			if space_zh and lang == 'zh':
				new_string = spliteKeyWord(cxt)
				# print(cxt)
				# print(' '.join(new_string))
				cxt=' '.join(new_string)
			if use_xlmr_tokenization:
				if length_limit - subtoken_length_count < 10:
					break
			if cxt_id>max_idx*2:
				break
			cxt = ''.join(c for c in cxt if c.isprintable())
			if lang is not None and lang == 'zh':
				split_seq = cxt.split()
			else:
				split_seq = cxt.split()
			if for_luke:
				if is_conll:
					words=[word + ' O O O' for word in split_seq]
				else:
					words=[word + ' O' for word in split_seq]
			else:
				if is_conll:
					words=[word + ' B-X B-X B-X' for word in split_seq]  B-X
				else:
					words=[word + ' B-X' for word in split_seq]
			# words=[word + ' O O B-X' for word in cxt.split()]
			if use_xlmr_tokenization:
				cxt_length = len(tokenizer.tokenize(' '.join(split_seq)))
				# if lang is not None and lang == 'zh' and subtoken_length_count>470:
				#   temp_sent=start_word+sentence+[eos]+end_word + words
				#   if len(tokenizer.tokenize(' '.join(temp_sent))) > length_limit:
				#       continue
				if cxt_length + subtoken_length_count + add_eos > length_limit:
					continue
				subtoken_length_count+=cxt_length
				used_context.append(cxt)
				used_context_length.append(cxt_length)
				#res = [len(tokenizer.tokenize(x)) for x in used_context]
				# res=[]
				# res+=tokenizer.tokenize(temp_keyword)+[tokenizer._eos_token]
				# for x in used_context: res+=tokenizer.tokenize(x)
			if len(words) + len(start_word) + len(end_word) + len(sentence) + add_eos > length_limit:
				break
			current_context_count+=1
			# =============================== before + after ====================
			if not add_eos:
				if cxt_id<max_idx:
					start_word = start_word + words
					if add_eos:
						start_word.append(eos)
				else:
					end_word = end_word + words
					if add_eos:
						end_word.append(eos)
			else:
				# pdb.set_trace()
				# =============================== eos ====================
				end_word = end_word + words
				# =============================== eos ====================
		# if add_eos and len(end_word)>0:
		#   sentence.append(eos)
		num_context_for_doc.append(current_context_count)
		if clean_file:
			new_sentence=sentence
		elif add_eos and len(end_word) > 0:
			if for_luke:
				new_sentence=['-DOCSTART- O']+['']+sentence+['']+[eos]+end_word
			else:
				new_sentence=start_word+sentence+[eos]+end_word
		else:
			if for_luke:
				new_sentence=['-DOCSTART- O']+['']+sentence+['']+end_word
			else:
				new_sentence=start_word+sentence+end_word
		# new_sentence=sentence
		if new_sentence[-1] == eos and add_eos:
			new_sentence = new_sentence[:-1]
		# new_sentence=sentence
		new_sentences.append(new_sentence)
		if use_xlmr_tokenization:
			# if lang is not None and lang == 'zh':
			#   keyword=temp_keyword+' '.join([word.split()[0] for word in [eos]+end_word])
			# else:
			keyword=' '.join([word.split()[0] for word in new_sentence])
			eos_token = None
			if tokenizer._eos_token is not None:
				keyword = re.sub('<EOS>', tokenizer._eos_token, keyword)
			else:
				keyword = re.sub('<EOS>', tokenizer._sep_token, keyword)
			temp_subtoken = tokenizer.tokenize(keyword)
			if len(temp_subtoken)>length_limit:
				pdb.set_trace()
	print(f'averaged number of context per sentence: {sum(num_context)/len(num_context)}')
	if sum(num_context) == 0:
		return new_sentences, failed_dict
	print(f'number of sentences have contexts: {(np.array(num_context)>=1).sum()}/{len(num_context)}')
	print(f'Number of sentences that all context are removed: {removed_context}')
	print(f'averaged number of context per DOC sentence: {sum(num_context_for_doc)/len(num_context_for_doc)}')
	print(f'distribution number of context per DOC sentence: {sum([x<=20 for x in num_context_for_doc])}, {sum([x>20 for x in num_context_for_doc])}, {max(num_context_for_doc)}')
	print(f'Max sent length: {max([len(x) for x in new_sentences])}')
	return new_sentences, failed_dict


def get_named_entity(sentence,is_bio=True, is_conll=False):
	entity_dict = {}
	named_entity = []
	correspond_label = []
	for word in sentence:
		try:
			if is_conll:
				entity, pos_tag, chunk, label = word.split()
			else:
				entity, label = word.split()
		except:
			# res = word.split()
			# entity, label = res[0], res[-1]
			pdb.set_trace()
		if label.startswith('B-'):
			if len(named_entity) != 0:
				entity_dict[' '.join(named_entity).lower()] = correspond_label
				named_entity = []
				correspond_label = []
			named_entity.append(entity)
			correspond_label.append(label)
		elif label.startswith('O'):
			if len(named_entity) != 0:
				entity_dict[' '.join(named_entity).lower()] = correspond_label
			named_entity = []
			correspond_label = []
		elif label.startswith('I-'):
			named_entity.append(entity)
			correspond_label.append(label)
		else:
			pdb.set_trace()
	if len(named_entity) != 0:
		entity_dict[' '.join(named_entity).lower()] = correspond_label
		named_entity = []
		correspond_label = []
	return entity_dict

def match_entity_count(named_entity_dict, context)-> int:
	score = 0
	context = context.lower()
	for entity in named_entity_dict:
		if entity in context:
			res = [m.start() for m in re.finditer(entity, context)]
			score += len(res)
	return score


def context_ranking(bow, context_list, origin_sent, removed_context = 0) -> list:
	score_list=[]
	# pdb.set_trace()
	for context in context_list:
		context = context.lower()
		cxt_bow = set(context.split())
		cxt_bow = cxt_bow - (cxt_bow & en_stopwords)
		score = len(bow & cxt_bow)/len(bow | cxt_bow)
		score_list.append(score)
		# score_list.append(match_entity_count(named_entity_dict, context))
	zipped_lists = zip(score_list, context_list)
	sorted_context_list = [y[1] for y in sorted(zipped_lists, reverse=True) if y[0]>0]
	# ==========debug===========
	# zipped_lists = zip(score_list, context_list)
	# ranking_list = [y for y in sorted(zipped_lists, reverse=True)]
	# print(sorted_context_list)
	# print(context_list)
	# print(ranking_list)
	# print(origin_sent)
	# print(set(context_list) - set(sorted_context_list))
	# pdb.set_trace()
	if len(context_list)>0 and len(sorted_context_list)==0:
		removed_context+=1
		# pdb.set_trace()
	return sorted_context_list, removed_context

def unlabeled_assignment(sentences,google_dict,failed_dict):
	new_sentences=[]
	for sentence in sentences:
		keyword=' '.join([word.split()[0] for word in sentence])
		# keyword = sentence
		if keyword not in google_dict:
			failed_dict.append(keyword)
			# new_sentences.append(sentence)
			continue
		named_entity_dict = get_named_entity(sentence)
		context = google_dict[keyword]
		start_word=[]
		end_word=[]
		for cxt_id, cxt in enumerate(context):
			# words=[word for word in cxt.split()]
			matched_entities = []
			for entity in named_entity_dict:
				if entity in cxt.lower():
					matched_entities.append(entity)
			if len(matched_entities) == 0:
				continue
			words = cxt.split()
			# words = nltk.word_tokenize(cxt)
			if len(words) > 100:
				continue
			# cxt = ' '.join(words)
			spans=[]
			correspond_label = []
			potential_spans = []
			correspond_potential_index = []
			for i in range(len(words)):
				for j in range(i+1,len(words)+1):
					span = words[i:j]
					span_word = ' '.join(span).lower()
					potential_spans.append(span_word)
					correspond_potential_index.append(list(range(i,j)))
			for entity in matched_entities:
				indices = [i for i,x in enumerate(potential_spans) if x == entity]
				for index in indices:
					spans.append(correspond_potential_index[index])
					correspond_label.append(named_entity_dict[potential_spans[index]])
				# for i in range(len(words)):
				#   for j in range(i+1,len(words)):
				#       span = words[i:j]
				#       span_word = ' '.join(span).lower()
				#       if span_word == entity:
				#          range(i,j)])
				#           spans.append(list(range(i,j)))
				#           correspond_label.append(named_entity_dict[span_word])
			if len(spans) == 0:
				continue
			else:
				annotated_words = []
				for span in spans:
					annotated_words += span
				if len(set(annotated_words))!=len(annotated_words):
					# pdb.set_trace()
					print(sentence)
					print(cxt)
					print('==============================')
					continue
				for spanid, span in enumerate(spans):
					for pos_id, index in enumerate(span):
						words[index] = words[index] + ' ' + correspond_label[spanid][pos_id]
				non_entitys = set(range(len(words))) - set(annotated_words)
				if len(non_entitys) == len(words):
					pdb.set_trace()
				for idx in non_entitys:
					# words[idx] = words[idx] + ' B-X'
					words[idx] = words[idx] + ' B-X'
					# words[idx] = words[idx] + ' X'
				new_sentences.append(words)
			# print(matched_entities)
			# print(words)
			# pdb.set_trace()
		# pdb.set_trace()
		# new_sentence=start_word+sentence+end_word
		# new_sentence=sentence
		# new_sentences.append(new_sentence)
	return new_sentences, failed_dict


def write_file(to_write,sentences,max_len=100):
	write_file=to_write
	label = ' _ _ O'
	writer=open(write_file,'w')
	remove_count=0
	for sentence in sentences:
		if len(sentence)>max_len:
			remove_count+=1
			continue
		i = 0
		for word in sentence:
			if i == 0:
				if ' ' in word:
					word = word.replace(' ', label+'\n')
				writer.write(word+label+'\n')
				i = 1
			else:
				writer.write(word+'\n')

		writer.write('\n')
	writer.close()
	print(remove_count)


def filter_context_sentences(sentences):
	new_sentences = []
	for sentence in sentences:
		flag = 0
		for token in sentence:
			if 'B-X' in token and '<EOS>' in token:
				flag = 1
				break
		if flag:
			new_sentences.append(sentence)
	return new_sentences

def add_to_dict(has_rank,desc, rank, keyword, google_dict, wrong_count, chunked_count, flag, chunk_num = 999, lang = None, lang_dict = None):
	# if 'github' in desc.lower() or 'NLP' in desc or 'sequence labeling' in desc.lower() or 'pytorch' in desc.lower() or 'tensorflow' in desc.lower() or desc == '':
	#   return google_dict, wrong_count, chunked_count
	if u'\u200b' in desc:
		desc = re.sub(u'\u200b', '', desc)
	if chunk_num <999:
		words = nltk.word_tokenize(desc)
		if len(words)>chunk_num:
			words = words[:chunk_num]
			chunked_count+=1
		desc = ' '.join(words)
		for word in words:
			# if len(word)>20:
			#   desc = ' '.join(desc.split()[:20])
			#   wrong_count+=1
			if len(word)>chunk_num:
				flag=1
				# pdb.set_trace()
				wrong_count+=1
				break
	if flag:
		return google_dict, wrong_count, chunked_count
	if keyword not in google_dict:
		google_dict[keyword]=set()

	if has_rank:
		google_dict[keyword].add((int(rank), desc))
	else:
		google_dict[keyword].add(desc)
	if lang_dict is not None and lang is not None:
		lang_dict[keyword] = lang
	return google_dict, wrong_count, chunked_count




def concate(args):
	use_xlmr_tokenization = True
	tokenizer=''
	if use_xlmr_tokenization:
		from transformers import AutoTokenizer
		tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')

	lang_dist = []
	lang_map = []

	f = codecs.open(args.retrieval_file, 'r', errors='ignore')
	google_data = f.readlines()

	use_mixed_strategy = False
	is_wiki_retrieval = True
	use_sentence = args.use_sentence
	use_paragraph_entity = args.use_paragraph_entity
	query = None
	sentence = None
	paragraph = None

	f.close()
	failed_list = []
	google_dict = {}
	lang_dict = {}
	wrong_count = 0
	chunked_count = 0
	failed_lines = 0
	failed_entry = []
	check_wiki_dict = {}
	check_wiki_count_dict = {}
	link_dict = {}
	image_keywords = []
	lang_idx = 0
	for lineid, data in enumerate(google_data):
		if data == '.\n' or data == ',\n':
			continue
		if is_wiki_retrieval and '\t\t' in data:
			data = re.sub('\t\t', '\t', data)
		res = data.strip().split('\t') 利用'\t'分割
		# if len(res) == 1 and is_wiki_retrieval and data.endswith(',','.',):
		# 	wrong_count+=1
		# 	res=data.strip('\n').split('\t')
		if len(res) == 9 or len(res) == 11 or len(res) == 12:
			has_rank = True
		# pdb.set_trace()
		elif len(res) == 1 and is_wiki_retrieval:  #
			keyword = res[0]
			has_rank = True
			rank = 0
			continue
		elif len(res) == 2 and is_wiki_retrieval:
			keyword = res[0]
			has_rank = True
			rank = 0
			continue
		# elif (len(res) == 4 or re.match('.*set\d+_window\d+_.*',data) is not None) and is_wiki_retrieval:
		# 	if len(res)!=4:
		# 		pdb.set_trace()
		# 	keyword = res[3]
		# 	google_dict[keyword]=set()
		# 	has_rank = True
		# 	rank = 0
		# 	continue
		elif len(res) == 0 and is_wiki_retrieval:
			keyword = None
			rank = 0
		elif not is_wiki_retrieval:
			has_rank = False
			failed_lines += 1
			failed_entry.append(res)
			continue
		# continue
		if len(res) == 0:
			continue
		if len(res) == 1 and not is_wiki_retrieval:
			# pdb.set_trace()
			print(lineid)
			wrong_count += 1
			continue
		if is_wiki_retrieval and len(res) != 6:
			if len(res) < 6:
				failed_list.append(data)
				continue
				pdb.set_trace()
			sentence = res[0]
			match = res[-1]
			link = res[-2]
			score = res[-3]
			title = res[-4]
			desc = ' '.join(res[1:-4])
			res = [sentence, desc, title, score, link, match]
		# pdb.set_trace()
		# if is_wiki_retrieval and len(res) != 5:
		# 	if len(res)<5:
		# 		failed_list.append(data)
		# 		continue
		# 		pdb.set_trace()
		# 	sentence=res[0]
		# 	# match=res[-1]
		# 	link=res[-1]
		# 	score=res[-2]
		# 	title=res[-3]
		# 	desc = ' '.join(res[1:-3])
		# 	res=[sentence,desc,title,score,link]
		# 	# pdb.set_trace()
		if len(res) == 9:
			# eng
			# datetime,  desc,  keyword,  link,  num,  rank,  taskid,  title,  pt = res
			# language = 'en'
			# title,  desc,  rank,  keyword,  category,  link,  datetime,  taskid,  pt = res
			# language = 'zh'
			title, desc, _, _, meta, link, _, _, _ = res
			meta = json.loads(meta)
			image_id = meta["img_name"]
			rank = meta["rank"]
			keyword = image_id
			language = 'en'
			image_keywords.append(meta['key'])
		# language = 'zh'
		elif len(res) == 11:
			# new
			category, language, num, keyword, rank, title, desc, link, datetime, taskid, pt = res
		# elif len(res) == 5 and is_wiki_retrieval:
		# 	sentence, desc, title, score, link = res
		# 	title = "[ "+title+" ]"
		# 	language='unk'
		# 	if use_paragraph_entity:
		# 		sentence = match_origin_paragraph(sentence,desc)
		# 	if use_sentence:
		# 		sentence = title+' '+sentence
		# 		desc = title+' '+desc
		# 	rank+=1
		elif len(res) == 6 and is_wiki_retrieval: 走这里
			sentence, desc, title, score, link, match = res
			title = "[ " + title + " ]"
			language = 'unk'
			if use_paragraph_entity:
				sentence = match_origin_paragraph(sentence, desc)
			if use_sentence:
				sentence = title + ' ' + sentence
				desc = title + ' ' + desc
			rank += 1
		else:
			category, language, num, keyword, rank, title, url, desc, link, datetime, taskid, pt = res
		# if language not in check_wiki_dict:
		#   check_wiki_dict[language] = 0
		#   check_wiki_count_dict[language] = {}
		# if link not in link_dict:
		#   link_dict[link]=0
		# if 'start=' in link.lower():
		#   start_page = int(re.match('.*start=(\d+)',link.lower()).group(1))
		#   if keyword not in check_wiki_count_dict[language]:
		#       check_wiki_count_dict[language][keyword]=[]
		#   check_wiki_dict[language]+=1
		#   check_wiki_count_dict[language][keyword].append(start_page)
		# if 'wikipedia' in url.lower():
		#   if keyword not in check_wiki_count_dict[language]:
		#       check_wiki_count_dict[language][keyword]=[]
		#   check_wiki_dict[language]+=1
		#   check_wiki_count_dict[language][keyword].append(int(rank))
		flag = 0
		try:
			if use_sentence:
				google_dict, wrong_count, chunked_count = add_to_dict(has_rank,sentence, rank, keyword, google_dict, wrong_count,
																	  chunked_count, flag, lang=language,
																	  lang_dict=lang_dict)
			else:
				google_dict, wrong_count, chunked_count = add_to_dict(has_rank,desc, rank, keyword, google_dict, wrong_count,
																	  chunked_count, flag, lang=language,
																	  lang_dict=lang_dict)
			if not is_wiki_retrieval or (is_wiki_retrieval and not use_sentence):
				google_dict, wrong_count, chunked_count = add_to_dict(has_rank, title, rank, keyword, google_dict, wrong_count,
																	  chunked_count, flag, lang=language,
																	  lang_dict=lang_dict)
		except:
			pdb.set_trace()

	print(f'failed_lines: {failed_lines}')
	lower_count = 0
	temp_failed_dict = []
	# old_google_dict = google_dict.copy()
	if has_rank:
		# pdb.set_trace()
		for key in google_dict:
			google_dict[key] = list(set(google_dict[key]))
			google_dict[key] = [y[1] for y in sorted(google_dict[key])]
			if len(google_dict[key]) < 6:
				temp_failed_dict.append(key)
				lower_count += 1

	print(f"There are {lower_count} sentences have less than 6 retrieve texts")

	failed_dict = []
	mis_seped_sentences = []
	length_limit = 510
	# dataset_names= {'DE-German':'de', 'ES-Spanish':'es', 'NL-Dutch':'nl', 'RU-Russian':'ru','ZH-Chinese':'zh','KO-Korean':'ko', 'MIX_Code_mixed':'mix', 'TR-Turkish':'tr', 'HI-Hindi':'hi','FA-Farsi':'fa','EN-English':'en','BN-Bangla':'bn'}
	all_sentences = []
	failed_id = 0

	to_write, train_sentences = count_file('span.txt', 'train', is_file=True, comment_label='# id')
	print("finish")
	# to_write,dev_sentences=count_file(args.conll_folder+'/'+args.lang+'_dev.conll','train',is_file=True, comment_label = '# id')
	# to_write,test_sentences=count_file(args.conll_folder+'/'+args.lang+'_test.conll','train',is_file=True, comment_label = '# id')

	dataset_name = args.conll_folder
	orig_dataset_name = dataset_name
	dataset_name += '_conll_rank_eos'
	is_conll = True
	new_train_sentences, failed_dict = process_google(use_xlmr_tokenization,train_sentences, google_dict, failed_dict, is_conll=is_conll,
													  clean_file=False, full_doc=True, add_eos='eos' in dataset_name,
													  length_limit=length_limit, for_luke='luke' in dataset_name,
													  lang=args.lang, is_wiki_retrieval=is_wiki_retrieval, tokenizer=tokenizer)
	# new_dev_sentences, failed_dict = process_google(dev_sentences,google_dict,failed_dict, is_conll = is_conll, clean_file = False, full_doc = True, add_eos = 'eos' in dataset_name, length_limit = length_limit, for_luke = 'luke' in dataset_name, lang = args.lang, is_wiki_retrieval = is_wiki_retrieval)
	# new_test_sentences, failed_dict = process_google(test_sentences,google_dict,failed_dict, is_conll = is_conll, clean_file = False, full_doc = True, add_eos = 'eos' in dataset_name, length_limit = length_limit, for_luke = 'luke' in dataset_name, lang = args.lang, is_wiki_retrieval = is_wiki_retrieval)
	suffix = ''
	if args.use_sentence:
		suffix += '_sentence'
	if args.use_paragraph_entity:
		suffix += '_withent'
	write_dir = dataset_name + '_doc_full_wiki_v3' + suffix
	if not os.path.exists(write_dir):
		os.mkdir(write_dir)

	write_file(write_dir + '/train.txt', new_train_sentences, max_len=length_limit)
	# write_file(write_dir + '/train.txt', new_train_sentences, max_len=length_limit)
	# write_file(write_dir+'/dev.txt',new_dev_sentences,max_len=999)
	# write_file(write_dir+'/test.txt',new_test_sentences,max_len=999)
	# all_sentences.append(new_test_sentences)
	all_sentences.append(new_train_sentences)
