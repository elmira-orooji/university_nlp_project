import re
import os
import nltk
import math
import random
import numpy as np
from nltk.stem import PorterStemmer
from collections import Counter
import json
from collections import defaultdict
from datetime import datetime

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Dataset')
tp_path = os.path.join(DATASET_PATH, 'TextProcessing', '61085-0.txt')
tc_path = os.path.join(DATASET_PATH, 'Classification-Train And Test')
ir_path = os.path.join(DATASET_PATH, 'IR', 'IR Documents to Index')


#-------------------------------------------------------Menues & get input----------------------------------------------------------------------

def menu():
    print('------------------ Main menu ---------------------\n')
    print('1- Text processing')
    print('2- Text classification')
    print('3- Information retrival')
    print('4- Exit')
    print('----------------------------------------------------')
    get_user_input()


def get_user_input(): 
     while True:
         try:
            user_input = int(input("Please enter a number between 1-4 :"))
            if( user_input == 1 ):
             text_processing_menu()
            elif( user_input == 2 ):
             text_classification_menu()
            elif( user_input == 3 ):
             IR_menu()
            elif( user_input == 4):
             exit()
            else: 
              print("*** Invalid number , try again please! ***")
              user_input = int(input("Please enter a number between 1-4 :"))
         except ValueError:
              print("*** Invalid number , try again please! ***")
     
          



def text_processing_menu():
    print('----------------- Text processing menu ------------------\n')
    print('1- Tokenize ')
    print('2- Lowercase folding')
    print('3- Token frequency')
    print('4- Stemming')
    print('5- Stop word removal')
    print('6- Back')
    print('7- Exit')
    print('-----------------------------------------------------------')
    get_tp_input()


def get_tp_input():
    while True:
         try:
            tp_input = int(input("Please enter a number between 1-6 :"))
            if( tp_input == 1 ):
             tp_tokenize(tp_path)
            elif( tp_input == 2 ):
             tp_lowercase_folding(tp_path)
            elif( tp_input == 3 ):
             tp_token_frequency(tp_path)
            elif( tp_input == 4):
             tp_stemming(tp_path)
            elif( tp_input == 5):
             tp_stop_word_removal(tp_path)
            elif( tp_input == 6):
             menu()
            elif( tp_input == 7):
             exit()
            else: 
              print("*** Invalid number , try again please! ***")
              tp_input = int(input("Please enter a number between 1-6 :"))
         except ValueError:
              print("*** Invalid number , try again please! ***")






def text_classification_menu():
    print('--------------- Text classification menu ----------------------\n')
    print('1- Naive bayes probability')
    print('2- Back')
    print('3- Exit')
    print('------------------------------------------------------------------')
    get_tc_input()


def get_tc_input():
    while True:
         try:
            tc_input = int(input("Please enter a number between 1-3 :"))
            if( tc_input == 1 ):
             evaluate_with_correct_structure()
            elif( tc_input == 2 ):
             menu()
            elif( tc_input == 3):
             exit()
            else: 
              print("*** Invalid number , try again please! ***")
              tc_input = int(input("Please enter a number between 1-3 :"))
         except ValueError:
              print("*** Invalid number , try again please! ***")
    





def IR_menu():
    print('--------------- Information retrival menu ----------------------\n')
    print('1- TF-IDF model ')
    print('2- Vector space model ')
    print('3- Back')
    print('4- Exit')
    print('------------------------------------------------------------------')
    get_IR_input()


def get_IR_input():
    while True:
         try:
            ir_input = int(input("Please enter a number between 1-4 :"))
            if( ir_input == 1 ):
             run_tfidf_ir_system_with_user_query()
            elif( ir_input == 2 ):
             run_vsm_ir_system_with_user_query()
            elif( ir_input == 3 ):
             menu()
            elif( ir_input == 4):
             exit()
            else: 
              print("*** Invalid number , try again please! ***")
              ir_input = int(input("Please enter a number between 1-4 :"))
         except ValueError:
              print("*** Invalid number , try again please! ***")
        


#---------------------------------------------------------1- Start text processing section--------------------------------------------------------------------------------------------


def tp_tokenize(filename):
    patterns = [ r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', r'(https?:\/\/|www\.)[a-zA-Z0-9./?#&=\-_%]+',
                r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}', r'\d{1,2}:\d{2}(:\d{2})?(\s*[AP]M)?',
                r'\d+\.\d+(e[+-]?\d+)?|\d+e[+-]?\d+', r'\d{1,3}(,\d{3})+', r'\d+', r"[A-Za-z]+'[A-Za-z]+|'[A-Za-z]+|[A-Za-z]+'", 
                r'#[A-Za-z0-9_]+|@[A-Za-z0-9_]+', r'\d+(\.\d+)?\s*(kg|g|lb|oz|km|m|cm|mm|°C|°F)', 
                r'[A-Za-z]+-[A-Za-z]+(-[A-Za-z]+)*|[A-Z][a-z]+[A-Z][a-z]+', r'[A-Za-z]+', r'\b[A-Za-z]\b', r'[.,!?;:()\[\]{}"\'\-–—]', r'[^\s]'
               ]
    
    pattern = '|'.join(f'({p})' for p in patterns)

    try:
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()
        

        tokens = []
        for match in re.finditer(pattern, text, re.IGNORECASE):
            for group in match.groups():
                if group is not None:
                    token = group.strip()
                    if token: 
                        tokens.append(token)
                    break
            
        with open('Tokenize_result.txt', 'w', encoding='utf-8') as file:
            file.write('\n'.join(tokens))
        
        print("** Successfully tokenized , Check the directory to see Tokenize_result file **")
        
        return tokens
    
    except FileNotFoundError:
        print(f"Error: File not found!")
    except Exception as e:
        print(f"Error: {e}")



def tp_lowercase_folding(filename):
   try:
    with open(filename , 'r' , encoding='utf-8') as file:
       content = file.read()
   
    lowercase_content = content.lower()

    with open('Lowercase_result.txt' , 'w' , encoding='utf-8') as file:
       file.write(lowercase_content)

    print("** Successfully lowercased , Check the directory to see lowercase_result file **")

   except FileNotFoundError:
     print(f"Error: File not found!")
   except Exception as e:
        print(f"Error: {e}")



def tp_token_frequency(filename):
   try:
    tp_tokens = tp_tokenize(filename)
    tk_counts = Counter(tp_tokens)

    with open('Token_frequncy_result.txt' , 'w' , encoding='utf-8') as file:
       for token,count in tk_counts.items():
           file.write(f"{token} : {count} \n")
         
    print("** Successfully Counted , Check the directory to see Token_frequency_result file **")

   except FileNotFoundError:
     print(f"Error: File not found!")
   except Exception as e:
        print(f"Error: {e}")



def tp_stemming(filename):
    with open(filename , 'r' , encoding='utf-8') as file:
       content = file.read()

    stemmer = PorterStemmer()
    words = content.split()
    stemmed_words = [stemmer.stem(word) for word in words]

    with open('stemmming_result.txt' , 'w' , encoding='utf-8') as file:
        file.write(" ".join(stemmed_words))

    print("** Successfully Stemmed , Check the directory to see stemming_result file **")



def tp_stop_word_removal(filename):
    stop_words = set([
    'a', 'an', 'the','i', 'me', 'my', 'myself', 'we', 'us', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves','what', 'which', 'who', 'whom', 'whose','ninth', 'tenth',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself','and', 'but', 'or', 'nor', 'for', 'so', 'yet',
    'although', 'because', 'since', 'unless', 'until', 'when', 'whenever','about', 'above', 'across', 'after', 'against',
    'along', 'among', 'around','at', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond',
    'where', 'whereas', 'wherever', 'whether', 'while','by', 'down', 'during', 'except', 'for', 'from', 'in', 'inside', 'into',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves','this', 'that', 'these', 'those',
    'near', 'of', 'off', 'on', 'onto', 'out', 'outside', 'over', 'past','up', 'upon', 'with', 'within', 'without',
    'through', 'throughout', 'to', 'toward', 'under', 'underneath', 'until', 'as', 'at', 'by', 'for', 'in', 'of', 'on', 'per', 'to', 'via',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such','can', 'will', 'just', 'don', 'should', 'now',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very','will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing','am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'now', 'then', 'when', 'always', 'never', 'often', 'sometimes', 'usually','here', 'there', 'where', 'why', 'how','very', 'too', 'so', 'just', 'really', 'quite', 'rather', 'almost',
    'also', 'even', 'just', 'only', 'simply','mr', 'mrs', 'ms', 'dr', 'prof', 'st', 'etc', 'eg', 'ie', 'vs',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday','one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth','january', 'february', 'march', 'april', 'may', 'june', 'july',
    'august', 'september', 'october', 'november', 'december','spring', 'summer', 'autumn', 'fall', 'winter','about', 'above', 'across', 'after', 'again', 'against', 'all', 'almost',
    'alone', 'along', 'already', 'also', 'although', 'always', 'among', 'and','another', 'any', 'anybody', 'anyone', 'anything', 'anywhere', 'are',
    'around', 'as', 'at', 'back', 'be', 'because', 'been', 'before', 'being','below', 'between', 'both', 'but', 'by', 'can', 'cannot', 'come', 'could',
    'did', 'do', 'does', 'done', 'down', 'during', 'each', 'either', 'enough', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere',
    'except', 'few', 'first', 'for', 'from', 'further', 'get', 'give', 'go', 'had', 'has', 'have', 'he', 'hence', 'her', 'here', 'hers', 'herself',
    'him', 'himself', 'his', 'how', 'however', 'i', 'if', 'in', 'indeed', 'into', 'is', 'it', 'its', 'itself', 'just', 'keep', 'last', 'least',
    'less', 'let', 'like', 'likely', 'made', 'make', 'many', 'may', 'me', 'might', 'more', 'most', 'much', 'must', 'my', 'myself', 'need', 'neither',
    'never', 'next', 'no', 'nobody', 'none', 'nor', 'not', 'nothing', 'now','nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto',
    'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please', 'put', 'rather', 're',
    'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'several', 'she','should', 'since', 'so', 'some', 'somebody', 'someone', 'something',
    'someday', 'somehow', 'somewhat', 'somewhere', 'still', 'such', 'take','than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then',
    'there', 'therefore', 'these', 'they', 'thing', 'things', 'think', 'this','those', 'though', 'through', 'thus', 'to', 'together', 'too', 'toward',
    'towards', 'try', 'under', 'until', 'up', 'upon', 'us', 'use', 'used', 'uses', 'very', 'want', 'wants', 'was', 'way', 'we', 'well', 'went',
    'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where','whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever',
    'whether', 'which', 'whichever', 'while', 'whither', 'who', 'whoever','whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without',
    'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves'
    ])
    
    with open(filename , 'r' , encoding='utf-8') as file:
       content = file.read()
    
    content = re.sub(r'<[^>]+>',"",content)
    content = re.sub(r'https?://\S+|wwww\.\S+',"",content)
    content = re.sub(pattern = r'['r'\U0001F600-\U0001F64F'r'\U0001F300-\U0001F5FF'r'\U0001F680-\U0001F6FF'r'\U0001F700-\U0001F77F'r']'
                      , repl = "",string = content, flags = re.UNICODE)
    

    words = re.findall(r'\b\w+\b',content.lower())
    filtered_words = [word for word in words if word not in stop_words]
    result = ' '.join(filtered_words)

    with open('stop_word_removal_result.txt' , 'w' , encoding='utf-8') as file:
        file.write(result)

    print("** Successfully Removed stop words , Check the directory to see stop_word_removal_result file **")



#---------------------------------------------------------End text processing section--------------------------------------------------------------------------------------------
    


#---------------------------------------------------------- Text Classification ------------------------------------------------------------------------

def extract_words( text, min_length = 3):
   
    words = []
    current_word = ""
    
    for char in text.lower():
        if 'a' <= char <= 'z':
            current_word += char
        else:
            if len(current_word) >= min_length:
                words.append(current_word)
            current_word = ""
    
    if len(current_word) >= min_length:
        words.append(current_word)
    
    return words

def log_to_prob(scores_dict):
    
    if not scores_dict:
        return {}
    
    scores = list(scores_dict.values())
    classes = list(scores_dict.keys())

    scores_array = np.array(scores, dtype=np.float64)
    
    max_score = np.max(scores_array)
   
    shifted_scores = scores_array - max_score
    
    exp_scores = np.exp(np.clip(shifted_scores, -100, 100))
    
    total = np.sum(exp_scores)
    
    if total == 0:
       
        return {cls: 1/len(classes) for cls in classes}
    
    probabilities = exp_scores / total
    
    probabilities = np.maximum(probabilities, 1e-10)
    
    probabilities = probabilities / np.sum(probabilities)
    
    return {cls: float(prob) for cls, prob in zip(classes, probabilities)}



def train_naive_bayes(base_path, max_files_per_class=100):
   
    classes = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            classes.append(item)
    
    if not classes:
        return None, None
    
    classes.sort()
    
    vocab = set()
    class_info = {}
    
    for cls in classes:
        class_path = os.path.join(base_path, cls)
        
        train_files = [f for f in sorted(os.listdir(class_path))
                      if os.path.isfile(os.path.join(class_path, f)) 
                      and f.lower().endswith('.txt')]
        
        if not train_files:
            print(f"  Warning: No training files for class '{cls}'")
            continue
        
        all_words = []
        files_processed = 0
        
        for file in train_files[:max_files_per_class]:
            file_path = os.path.join(class_path, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    words = extract_words(text)
                    all_words.extend(words)
                    for word in words:
                        vocab.add(word)
                    
                    files_processed += 1
            except OSError:
                continue
        
        if files_processed > 0:
            class_info[cls] = {
                'words': all_words,
                'word_counts': Counter(all_words),
                'file_count': files_processed,
                'total_words': len(all_words),
                'unique_words': len(set(all_words))
            }
            
            print(f"{cls}: {files_processed} training files - {len(all_words):} words")
    
    if not class_info:
        return None, None
    
    return class_info, sorted(vocab)

def classify_document(content, class_info, vocab):
    """Score known vocabulary only; return four values on every path."""
    if not class_info or not vocab:
        return None, None, None, None
    vocab_set = set(vocab)
    counts = Counter(word for word in extract_words(content) if word in vocab_set)
    if not counts:
        return None, None, None, None

    total_files = sum(info['file_count'] for info in class_info.values())
    if total_files <= 0:
        return None, None, None, None
    scores = {}
    for cls, info in class_info.items():
        # Models trained before word_counts was added remain compatible.
        training_counts = info.get('word_counts')
        if training_counts is None:
            training_counts = Counter(info['words'])
        if info['file_count'] <= 0:
            continue
        score = math.log(info['file_count'] / total_files)
        denominator = info['total_words'] + len(vocab_set)
        for word, frequency in counts.items():
            score += frequency * math.log((training_counts.get(word, 0) + 1) / denominator)
        scores[cls] = score

    if not scores:
        return None, None, None, None
    probabilities = log_to_prob(scores)
    predicted_class = max(scores, key=scores.get)
    return scores, probabilities, predicted_class, probabilities[predicted_class]


def evaluate_with_correct_structure():
    print("----------------Training general info------------------\n")
   
    if not os.path.exists(tc_path):
        print(f"Error: Dataset folder not found")
        return
    
    class_info, vocab = train_naive_bayes(tc_path, max_files_per_class=100)
    
    if not class_info or not vocab:
        print("Error: No training data found!")
        return
    
    vocab_size = len(vocab)
    print(f"Classes: {len(class_info)}")
    print(f"Vocabulary: {vocab_size:,} words\n")
    
    
    print("---------------------Test file------------------------\n" )
  
    all_test_files = []
    classes = list(class_info.keys())
    
    for cls in classes:
        test_folder = os.path.join(tc_path, cls, 'test')
        if os.path.exists(test_folder):
            test_files = [f for f in os.listdir(test_folder) 
                         if f.lower().endswith('.txt')]
            for file in test_files:
                all_test_files.append((cls, file))
    
    if not all_test_files:
        print("Error: No test files found!")
        return
    
    print(f"Total test files: {len(all_test_files):,}")
    
    true_class_folder, filename = random.choice(all_test_files)
    file_path = os.path.join(tc_path, true_class_folder, 'test', filename)
    
    print(f"Random test file selected:")
    print(f"Filename: {filename}")
    print(f"True class: {true_class_folder}")
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    first_line = content.split('\n')[0].strip().lower() if '\n' in content else content.strip().lower()
    true_class_content = None
    for cls in classes:
        if cls.lower() in first_line:
            true_class_content = cls
            break
    true_class = true_class_content if true_class_content else true_class_folder
    scores, probabilities, predicted_class, confidence = classify_document(
        content, class_info, vocab
    )
    
    if not scores:
        print("Error: Could not classify document!")
        return
    
    is_correct = (predicted_class == true_class)
    
    
    print(f"\n----------------Classification results-----------------\n")
    print(f"True class: {true_class}")
    print(f"Predicted class: {predicted_class}")
    print(f"Predict:{'CORRECT' if is_correct else 'WRONG'}")
    
    print(f"\n---------------Log probability--------------------------\n")
    for cls, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
       
        print(f"{cls:25s}: {score:12.2f}")
    
    print(f"\n-----------------Normalized probability-----------------\n")
    for cls, prob in sorted(probabilities.items()):
     
        
        if prob < 0.0001:
            prob_str = f"{prob:.2e}"
        else:
            prob_str = f"{prob:.2%}"
        
        print(f"{cls:25s}: {prob_str:>12}")
    
    max_score = max(scores.values())
    for cls in classes:
        diff = scores[cls] - max_score

    print("\n-------------------Evaluation system------------------\n")
 
    all_results = []
    confusion_matrix = {true_cls: {pred_cls: 0 for pred_cls in classes} for true_cls in classes}
    total_evaluated = 0
     
    for cls in classes:
        test_folder = os.path.join(tc_path, cls, 'test')
        if not os.path.exists(test_folder):
            continue
        
        test_files = [f for f in os.listdir(test_folder) 
                     if f.lower().endswith('.txt')][:50]  
        
        if not test_files:
            continue
        
        for file in test_files:
            file_path = os.path.join(test_folder, file)
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                _, _, pred_class, conf = classify_document(content, class_info, vocab)
                
                if pred_class:
                    is_correct_local = (pred_class == cls)
                    
                    all_results.append({
                        'true': cls,
                        'pred': pred_class,
                        'correct': is_correct_local,
                        'confidence': conf
                    })
                    
                    confusion_matrix[cls][pred_class] += 1
                    total_evaluated += 1
                    
            except OSError:
                continue
    
    if not all_results:
        print("Error: No test files could be evaluated!")
        return
    
    correct = sum(1 for r in all_results if r['correct'])
    accuracy = correct / total_evaluated if total_evaluated > 0 else 0
    
    print(f"Accuracy: {correct}/{total_evaluated} = {accuracy:.2%}")
    
    print(f"\n{'Class':25s} | {'Precision':10s} | {'Recall':10s} | {'F1':10s} | {'Support':8s}")
    print("-" * 75)
    
    macro_precision = []
    macro_recall = []
    
    for cls in classes:
        tp = confusion_matrix[cls][cls]
        fp = sum(confusion_matrix[other][cls] for other in classes if other != cls)
        fn = sum(confusion_matrix[cls][other] for other in classes if other != cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = tp + fn
        
        macro_precision.append(precision)
        macro_recall.append(recall)
        
        print(f"{cls:25s} | {precision:10.4f} | {recall:10.4f} | {f1:10.4f} | {support:8d}")
    
    macro_precision_avg = np.mean(macro_precision) if macro_precision else 0
    macro_recall_avg = np.mean(macro_recall) if macro_recall else 0
    macro_f1 = 2 * macro_precision_avg * macro_recall_avg / (macro_precision_avg + macro_recall_avg) \
               if (macro_precision_avg + macro_recall_avg) > 0 else 0
    
    print("-" * 75)
    print(f"{' AVERAGE':25s} | {macro_precision_avg:10.4f} | {macro_recall_avg:10.4f} | {macro_f1:10.4f} | {total_evaluated:8d}")


#----------------------------------------------------------------End Text Classification ------------------------------------------------------


    

#---------------------------------------------------------Start TF_IDF Section------------------------------------------------------------

def run_tfidf_ir_system_with_user_query(
    documents_path=None,
    output_base_dir="ir_results",
    max_documents=None
):
   
    if documents_path is None:
        documents_path = ir_path
        
   
    if not os.path.exists(documents_path):
        print(f"\nERROR: Path '{documents_path}' not found!")
        return None
    
  
    output_dir = os.path.join(output_base_dir)
    os.makedirs(output_dir, exist_ok=True)
  
    index_result = build_tfidf_index(
        documents_path=documents_path,
        output_dir=os.path.join(output_dir, "index"),
        max_files=max_documents
    )
    
    if index_result is None:
        print("\nERROR: Failed to build index!")
        return None
    

    search_results = get_user_queries_and_rank(
        indexer=index_result['indexer'],
        output_dir=os.path.join(output_dir, "searches")
    )
  
    print("\n" + "=" * 70)
    print("SYSTEM EXECUTION COMPLETED")
    print("=" * 70)
    
    print(f"\nSUMMARY:")
    print(f"  Documents indexed: {index_result['summary']['documents_count']:,}")
    print(f"  Vocabulary size: {index_result['summary']['vocabulary_size']:,}")
    print(f"  Queries processed: {len(search_results)}")
    print(f"  Output directory: {output_dir}")
    
    return {
        'index_result': index_result,
        'search_results': search_results,
        'output_dir': output_dir
    }


def build_tfidf_index(documents_path, output_dir, max_files=None):

    class Indexer:
        def __init__(self):
            self.documents = []
            self.vocabulary = []
            self.term_index = {}
            self.tfidf_matrix = []
            self.idf_values = {}
            self.doc_lengths = []
            self.doc_frequencies = {}
            self.stats = {
                'total_files': 0,
                'processed_files': 0,
                'failed_files': 0,
                'vocab_size': 0,
                'matrix_shape': (0, 0)
            }
    
    indexer = Indexer()
    
    text_files = []
    for root, dirs, files in os.walk(documents_path):
        dirs.sort()
        files.sort()
        for file in files:
            if file.lower().endswith('.txt'):
                text_files.append(os.path.join(root, file))
    
    if max_files is not None:
        text_files = text_files[:max_files]
    
    if not text_files:
        print("  ERROR: No text files found!")
        return None
    
    indexer.stats['total_files'] = len(text_files)
    
    processed_count = 0
    
    for i, filepath in enumerate(text_files):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            content_lower = content.lower()
            words = []
            current_word = ""
            
            for char in content_lower:
                if 'a' <= char <= 'z':
                    current_word += char
                else:
                    if len(current_word) >= 3:
                        words.append(current_word)
                    current_word = ""
            
            if len(current_word) >= 3:
                words.append(current_word)
            
            stopwords = set([
    'a', 'an', 'the','i', 'me', 'my', 'myself', 'we', 'us', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves','what', 'which', 'who', 'whom', 'whose','ninth', 'tenth',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself','and', 'but', 'or', 'nor', 'for', 'so', 'yet',
    'although', 'because', 'since', 'unless', 'until', 'when', 'whenever','about', 'above', 'across', 'after', 'against',
    'along', 'among', 'around','at', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond',
    'where', 'whereas', 'wherever', 'whether', 'while','by', 'down', 'during', 'except', 'for', 'from', 'in', 'inside', 'into',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves','this', 'that', 'these', 'those',
    'near', 'of', 'off', 'on', 'onto', 'out', 'outside', 'over', 'past','up', 'upon', 'with', 'within', 'without',
    'through', 'throughout', 'to', 'toward', 'under', 'underneath', 'until', 'as', 'at', 'by', 'for', 'in', 'of', 'on', 'per', 'to', 'via',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such','can', 'will', 'just', 'don', 'should', 'now',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very','will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing','am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'now', 'then', 'when', 'always', 'never', 'often', 'sometimes', 'usually','here', 'there', 'where', 'why', 'how','very', 'too', 'so', 'just', 'really', 'quite', 'rather', 'almost',
    'also', 'even', 'just', 'only', 'simply','mr', 'mrs', 'ms', 'dr', 'prof', 'st', 'etc', 'eg', 'ie', 'vs',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday','one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth','january', 'february', 'march', 'april', 'may', 'june', 'july',
    'august', 'september', 'october', 'november', 'december','spring', 'summer', 'autumn', 'fall', 'winter','about', 'above', 'across', 'after', 'again', 'against', 'all', 'almost',
    'alone', 'along', 'already', 'also', 'although', 'always', 'among', 'and','another', 'any', 'anybody', 'anyone', 'anything', 'anywhere', 'are',
    'around', 'as', 'at', 'back', 'be', 'because', 'been', 'before', 'being','below', 'between', 'both', 'but', 'by', 'can', 'cannot', 'come', 'could',
    'did', 'do', 'does', 'done', 'down', 'during', 'each', 'either', 'enough', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere',
    'except', 'few', 'first', 'for', 'from', 'further', 'get', 'give', 'go', 'had', 'has', 'have', 'he', 'hence', 'her', 'here', 'hers', 'herself',
    'him', 'himself', 'his', 'how', 'however', 'i', 'if', 'in', 'indeed', 'into', 'is', 'it', 'its', 'itself', 'just', 'keep', 'last', 'least',
    'less', 'let', 'like', 'likely', 'made', 'make', 'many', 'may', 'me', 'might', 'more', 'most', 'much', 'must', 'my', 'myself', 'need', 'neither',
    'never', 'next', 'no', 'nobody', 'none', 'nor', 'not', 'nothing', 'now','nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto',
    'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please', 'put', 'rather', 're',
    'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'several', 'she','should', 'since', 'so', 'some', 'somebody', 'someone', 'something',
    'someday', 'somehow', 'somewhat', 'somewhere', 'still', 'such', 'take','than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then',
    'there', 'therefore', 'these', 'they', 'thing', 'things', 'think', 'this','those', 'though', 'through', 'thus', 'to', 'together', 'too', 'toward',
    'towards', 'try', 'under', 'until', 'up', 'upon', 'us', 'use', 'used', 'uses', 'very', 'want', 'wants', 'was', 'way', 'we', 'well', 'went',
    'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where','whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever',
    'whether', 'which', 'whichever', 'while', 'whither', 'who', 'whoever','whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without',
    'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves'
    ])
            words = [w for w in words if w not in stopwords]
            
            filename = os.path.basename(filepath)
            relative_path = os.path.relpath(filepath, documents_path)
            
            doc_info = {
                'id': i,
                'filename': filename,
                'full_path': filepath,
                'relative_path': relative_path,
                'content_length': len(content),
                'word_count': len(words),
                'unique_words': len(set(words)),
                'words': words
            }
            
            indexer.documents.append(doc_info)
            processed_count += 1
            
        except Exception:
            continue
    
    if processed_count == 0:
        print("  ERROR: No files could be processed!")
        return None
    
    indexer.stats['processed_files'] = processed_count
    indexer.stats['failed_files'] = len(text_files) - processed_count
 
    all_words_set = set()
    doc_frequencies = defaultdict(int)
    
    for doc in indexer.documents:
        unique_words = set(doc['words'])
        all_words_set.update(unique_words)
        
        for word in unique_words:
            doc_frequencies[word] += 1
    
    indexer.vocabulary = sorted(all_words_set)
    indexer.term_index = {word: idx for idx, word in enumerate(indexer.vocabulary)}
    indexer.doc_frequencies = dict(doc_frequencies)
    indexer.stats['vocab_size'] = len(indexer.vocabulary)
    total_docs = len(indexer.documents)
    
    for word in indexer.vocabulary:
        doc_freq = doc_frequencies.get(word, 1)
        indexer.idf_values[word] = math.log(total_docs / doc_freq)
   
    vocab_size = len(indexer.vocabulary)
    
    for doc_idx, doc in enumerate(indexer.documents):
        words = doc['words']
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word] += 1
        
        tfidf_vector = [0.0] * vocab_size
        
        for word, count in word_counts.items():
            if word in indexer.term_index:
                idx = indexer.term_index[word]
                if count > 0:
                    tf = 1 + math.log(count)
                    idf = indexer.idf_values[word]
                    tfidf_vector[idx] = tf * idf
        
        indexer.tfidf_matrix.append(tfidf_vector)
        squared_sum = sum(w * w for w in tfidf_vector)
        doc_length = math.sqrt(squared_sum) if squared_sum > 0 else 1.0
        indexer.doc_lengths.append(doc_length)
        
    
    indexer.stats['matrix_shape'] = (len(indexer.tfidf_matrix), len(indexer.vocabulary))
    os.makedirs(output_dir, exist_ok=True)

    index_data = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'documents_path': documents_path,
            'total_documents': len(indexer.documents),
            'vocabulary_size': len(indexer.vocabulary),
            'matrix_shape': indexer.stats['matrix_shape']
        },
        'documents': [
            {
                'id': doc['id'],
                'filename': doc['filename'],
                'relative_path': doc['relative_path'],
                'word_count': doc['word_count']
            }
            for doc in indexer.documents
        ],
        'vocabulary_sample': indexer.vocabulary[:50],
        'idf_sample': {k: float(v) for k, v in list(indexer.idf_values.items())[:50]}
    }
    
    with open(os.path.join(output_dir, 'index_data.json'), 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2)
    
    matrix_sample = []
    for i in range(min(5, len(indexer.documents))):
        doc_weights = []
        for j in range(min(20, len(indexer.vocabulary))):
            weight = indexer.tfidf_matrix[i][j]
            if weight > 0:
                doc_weights.append({
                    'term': indexer.vocabulary[j],
                    'tfidf_weight': float(weight)
                })
        
        matrix_sample.append({
            'doc_id': i,
            'doc_name': indexer.documents[i]['filename'],
            'top_weights': doc_weights[:10]  
        })
    
    with open(os.path.join(output_dir, 'tfidf_matrix_sample.json'), 'w', encoding='utf-8') as f:
        json.dump(matrix_sample, f, indent=2)
    
    summary = {
        'total_files_found': indexer.stats['total_files'],
        'processed_files': indexer.stats['processed_files'],
        'failed_files': indexer.stats['failed_files'],
        'documents_count': len(indexer.documents),
        'vocabulary_size': len(indexer.vocabulary),
        'matrix_shape': indexer.stats['matrix_shape']
    }
    
    return {
        'indexer': indexer,
        'summary': summary,
        'index_dir': output_dir
    }


def get_user_queries_and_rank(indexer, output_dir):
     
    search_results = []
    os.makedirs(output_dir, exist_ok=True)
    query_count = 0
    
    while True:
        print("\n")
        
        query = input("\nEnter your query (or 'exit' to quit): ").strip()
        
        if query.lower() == 'exit':
            IR_menu()
            break
        
        if query.lower() == 'help':
            print("\nCommands:")
            print("  [any text] - Search query")
            print("  exit - Exit the system")
            print("  help - Show this help")
            continue
        
        if not query:
            print("Please enter a query.")
            continue
        
        query_count += 1
        print(f"\nQuery num{query_count}: '{query}'")
        ranked_results = rank_documents_with_tfidf(indexer, query)
        
        if not ranked_results:
            print("No relevant documents found.")
            continue
        
        print(f"\nTOP 10 RANKED RESULTS:")
        print("-" * 80)
        print(f"{'Rank':^6} | {'Score':^10} | {'Document':^40} | {'Word Count':^10}")
        print("-" * 80)
        
        for rank, result in enumerate(ranked_results[:10], 1):
            filename = result['filename']
            if len(filename) > 38:
                filename = filename[:35] + "..."
            
            print(f"{rank:^6} | {result['score']:^10.6f} | {filename:^40} | {result['word_count']:^10}")
        
        print("-" * 80)
     
        result_file = os.path.join(output_dir, f"query_{query_count:03d}.json")
        
        result_data = {
            'query': query,
            'query_number': query_count,
            'timestamp': datetime.now().isoformat(),
            'total_documents_found': len(ranked_results),
            'top_10_results': [
                {
                    'rank': i + 1,
                    'document_id': r['doc_id'],
                    'filename': r['filename'],
                    'score': float(r['score']),
                    'word_count': r['word_count']
                }
                for i, r in enumerate(ranked_results[:10])
            ]
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2)
        
        search_results.append({
            'query': query,
            'results': ranked_results,
            'file': result_file
        })
        
        print(f"\nSuccessfully result saved named : {os.path.basename(result_file)}")
    
    print(f"\nTotal queries processed: {query_count}")
    return search_results

def rank_documents_with_tfidf(indexer, query, top_k=None):
   
    query_lower = query.lower()
    query_words = []
    current_word = ""
    
    for char in query_lower:
        if 'a' <= char <= 'z':
            current_word += char
        else:
            if len(current_word) >= 3:
                query_words.append(current_word)
            current_word = ""
    
    if len(current_word) >= 3:
        query_words.append(current_word)
    
    if not query_words:
        return []
    
    query_vector = [0.0] * len(indexer.vocabulary)
    word_counts = {}
    
    for word in query_words:
        word_counts[word] = word_counts.get(word, 0) + 1

    for word, count in word_counts.items():
        if word in indexer.term_index:
            idx = indexer.term_index[word]
            if count > 0:
                tf = 1 + math.log(count)
                idf = indexer.idf_values.get(word, 0)
                query_vector[idx] = tf * idf
                print(f"'{word}': TF={tf:.3f}, IDF={idf:.3f}, TF-IDF={tf*idf:.3f}")
    
    results = []
    
    for doc_idx, doc in enumerate(indexer.documents):
        doc_vector = indexer.tfidf_matrix[doc_idx]
        doc_length = indexer.doc_lengths[doc_idx]
        dot_product = 0
        for i in range(len(query_vector)):
            dot_product += query_vector[i] * doc_vector[i]

        if dot_product == 0:
            continue
  
        query_squared_sum = sum(q * q for q in query_vector)
        if query_squared_sum == 0:
            continue
        
        query_length = math.sqrt(query_squared_sum)
        
        if doc_length > 0 and query_length > 0:
            similarity = dot_product / (query_length * doc_length)
            if similarity > 0:
                results.append({
                    'doc_id': doc_idx,
                    'filename': doc['filename'],
                    'relative_path': doc['relative_path'],
                    'score': similarity,
                    'word_count': doc['word_count']
                })
    
    results.sort(key=lambda x: x['score'], reverse=True)
    
    if top_k is not None:
        return results[:top_k]
    
    return results


def display_tfidf_matrix_info(indexer):
    print("\n" + "=" * 70)
    print("TF-IDF MATRIX INFORMATION")
    print("=" * 70)
    
    print(f"\nMatrix Dimensions: {len(indexer.tfidf_matrix)} documents × {len(indexer.vocabulary)} terms")
    print(f"Total Elements: {len(indexer.tfidf_matrix) * len(indexer.vocabulary):,}")
    
    non_zero_count = 0
    total_elements = len(indexer.tfidf_matrix) * len(indexer.vocabulary)
    
    for doc_vector in indexer.tfidf_matrix:
        for weight in doc_vector:
            if weight > 0:
                non_zero_count += 1
    
    sparsity = (total_elements - non_zero_count) / total_elements * 100
    print(f"Non-zero Elements: {non_zero_count:,} ({100-sparsity:.1f}%)")
    print(f"Sparsity: {sparsity:.1f}%")

    print(f"\nSample TF-IDF Weights (first 3 documents, first 5 terms):")
    print("-" * 60)
    
    for i in range(min(3, len(indexer.documents))):
        print(f"\nDocument {i}: {indexer.documents[i]['filename'][:30]}")
        for j in range(min(5, len(indexer.vocabulary))):
            weight = indexer.tfidf_matrix[i][j]
            if weight > 0:
                print(f"  {indexer.vocabulary[j]:15s}: {weight:.4f}")


#-------------------------------------------------------------END TF_IDF Section --------------------------------------------------------------



#---------------------------------------------------------Start VSM Section--------------------------------------------------------------------------

def run_vsm_ir_system_with_user_query(
    documents_path=None,
    output_base_dir="vsm_results",
    max_documents=None
):

    if documents_path is None:
        documents_path = ir_path

    if not os.path.exists(documents_path):
        print(f"\nERROR: Path '{documents_path}' not found!")
        return None
      
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    vsm_result = build_vector_space_model(
        documents_path=documents_path,
        output_dir=os.path.join(output_dir, "vsm_model"),
        max_files=max_documents
    )
    
    if vsm_result is None:
        print("\nERROR: Failed to build VSM!")
        return None
    
    search_results = get_user_queries_and_vsm_rank(
        vsm_model=vsm_result['vsm_model'],
        output_dir=os.path.join(output_dir, "searches")
    )

    print("\n" + "=" * 70)
    print("SYSTEM EXECUTION COMPLETED")
    print("=" * 70)
    
    print(f"\nSUMMARY:")
    print(f"  Documents indexed: {vsm_result['summary']['documents_count']:,}")
    print(f"  Vocabulary size: {vsm_result['summary']['vocabulary_size']:,}")
    print(f"  Queries processed: {len(search_results)}")
    print(f"  Output directory: {output_dir}")
    
    return {
        'vsm_result': vsm_result,
        'search_results': search_results,
        'output_dir': output_dir
    }


def build_vector_space_model(documents_path, output_dir, max_files=None):
    
    class VSMModel:
        def __init__(self):
            self.documents = []
            self.vocabulary = []
            self.term_index = {}
            self.document_vectors = []  
            self.doc_frequencies = {}
            self.stats = {
                'total_files': 0,
                'processed_files': 0,
                'failed_files': 0,
                'vocab_size': 0,
                'vector_dimensions': 0
            }
    
    vsm_model = VSMModel()
    text_files = []
    for root, dirs, files in os.walk(documents_path):
        dirs.sort()
        files.sort()
        for file in files:
            if file.lower().endswith('.txt'):
                text_files.append(os.path.join(root, file))
    
    if max_files is not None:
        text_files = text_files[:max_files]
    
    if not text_files:
        print("  ERROR: No text files found!")
        return None
    
    vsm_model.stats['total_files'] = len(text_files) 
    all_docs_words = []
    processed_count = 0
    
    for i, filepath in enumerate(text_files):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            content_lower = content.lower()
            words = []
            current_word = ""
            
            for char in content_lower:
                if 'a' <= char <= 'z':
                    current_word += char
                else:
                    if len(current_word) >= 3:
                        words.append(current_word)
                    current_word = ""
            
            if len(current_word) >= 3:
                words.append(current_word)
        
            stopwords = set([
    'a', 'an', 'the','i', 'me', 'my', 'myself', 'we', 'us', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves','what', 'which', 'who', 'whom', 'whose','ninth', 'tenth',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself','and', 'but', 'or', 'nor', 'for', 'so', 'yet',
    'although', 'because', 'since', 'unless', 'until', 'when', 'whenever','about', 'above', 'across', 'after', 'against',
    'along', 'among', 'around','at', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond',
    'where', 'whereas', 'wherever', 'whether', 'while','by', 'down', 'during', 'except', 'for', 'from', 'in', 'inside', 'into',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves','this', 'that', 'these', 'those',
    'near', 'of', 'off', 'on', 'onto', 'out', 'outside', 'over', 'past','up', 'upon', 'with', 'within', 'without',
    'through', 'throughout', 'to', 'toward', 'under', 'underneath', 'until', 'as', 'at', 'by', 'for', 'in', 'of', 'on', 'per', 'to', 'via',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such','can', 'will', 'just', 'don', 'should', 'now',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very','will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing','am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'now', 'then', 'when', 'always', 'never', 'often', 'sometimes', 'usually','here', 'there', 'where', 'why', 'how','very', 'too', 'so', 'just', 'really', 'quite', 'rather', 'almost',
    'also', 'even', 'just', 'only', 'simply','mr', 'mrs', 'ms', 'dr', 'prof', 'st', 'etc', 'eg', 'ie', 'vs',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday','one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth','january', 'february', 'march', 'april', 'may', 'june', 'july',
    'august', 'september', 'october', 'november', 'december','spring', 'summer', 'autumn', 'fall', 'winter','about', 'above', 'across', 'after', 'again', 'against', 'all', 'almost',
    'alone', 'along', 'already', 'also', 'although', 'always', 'among', 'and','another', 'any', 'anybody', 'anyone', 'anything', 'anywhere', 'are',
    'around', 'as', 'at', 'back', 'be', 'because', 'been', 'before', 'being','below', 'between', 'both', 'but', 'by', 'can', 'cannot', 'come', 'could',
    'did', 'do', 'does', 'done', 'down', 'during', 'each', 'either', 'enough', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere',
    'except', 'few', 'first', 'for', 'from', 'further', 'get', 'give', 'go', 'had', 'has', 'have', 'he', 'hence', 'her', 'here', 'hers', 'herself',
    'him', 'himself', 'his', 'how', 'however', 'i', 'if', 'in', 'indeed', 'into', 'is', 'it', 'its', 'itself', 'just', 'keep', 'last', 'least',
    'less', 'let', 'like', 'likely', 'made', 'make', 'many', 'may', 'me', 'might', 'more', 'most', 'much', 'must', 'my', 'myself', 'need', 'neither',
    'never', 'next', 'no', 'nobody', 'none', 'nor', 'not', 'nothing', 'now','nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto',
    'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please', 'put', 'rather', 're',
    'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'several', 'she','should', 'since', 'so', 'some', 'somebody', 'someone', 'something',
    'someday', 'somehow', 'somewhat', 'somewhere', 'still', 'such', 'take','than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then',
    'there', 'therefore', 'these', 'they', 'thing', 'things', 'think', 'this','those', 'though', 'through', 'thus', 'to', 'together', 'too', 'toward',
    'towards', 'try', 'under', 'until', 'up', 'upon', 'us', 'use', 'used', 'uses', 'very', 'want', 'wants', 'was', 'way', 'we', 'well', 'went',
    'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where','whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever',
    'whether', 'which', 'whichever', 'while', 'whither', 'who', 'whoever','whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without',
    'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves'
    ])
            words = [w for w in words if w not in stopwords]
            
            filename = os.path.basename(filepath)
            relative_path = os.path.relpath(filepath, documents_path)
            
            doc_info = {
                'id': i,
                'filename': filename,
                'full_path': filepath,
                'relative_path': relative_path,
                'content_length': len(content),
                'word_count': len(words),
                'unique_words': len(set(words)),
                'words': words
            }
            
            vsm_model.documents.append(doc_info)
            all_docs_words.append(words)
            processed_count += 1
            
        except Exception:
            continue
    
    if processed_count == 0:
        print("  ERROR: No files could be processed!")
        return None
    
    vsm_model.stats['processed_files'] = processed_count
    vsm_model.stats['failed_files'] = len(text_files) - processed_count
    
 
    all_words_set = set()
    doc_frequencies = defaultdict(int)
    
    for words in all_docs_words:
        unique_words = set(words)
        all_words_set.update(unique_words)
        
        for word in unique_words:
            doc_frequencies[word] += 1
    
    vsm_model.vocabulary = sorted(all_words_set)
    vsm_model.term_index = {word: idx for idx, word in enumerate(vsm_model.vocabulary)}
    vsm_model.doc_frequencies = dict(doc_frequencies)
    vsm_model.stats['vocab_size'] = len(vsm_model.vocabulary)
    vocab_size = len(vsm_model.vocabulary)
    total_docs = len(vsm_model.documents)
    idf_values = {}
    for word in vsm_model.vocabulary:
        doc_freq = doc_frequencies.get(word, 1)
        idf_values[word] = math.log(total_docs / doc_freq)
    
    for doc_idx, words in enumerate(all_docs_words):
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word] += 1
 
        tfidf_vector = [0.0] * vocab_size
        
        for word, count in word_counts.items():
            if word in vsm_model.term_index:
                idx = vsm_model.term_index[word]
                if count > 0:
                    tf = 1 + math.log(count) 
                    idf = idf_values[word]
                    tfidf_vector[idx] = tf * idf
        vsm_model.document_vectors.append(tfidf_vector) 

    vsm_model.stats['vector_dimensions'] = vocab_size
    vsm_model.idf_values = idf_values  
    
    os.makedirs(output_dir, exist_ok=True)
    
    vsm_data = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'documents_path': documents_path,
            'total_documents': len(vsm_model.documents),
            'vocabulary_size': len(vsm_model.vocabulary),
            'vector_dimensions': vocab_size
        },
        'documents': [
            {
                'id': doc['id'],
                'filename': doc['filename'],
                'relative_path': doc['relative_path'],
                'word_count': doc['word_count']
            }
            for doc in vsm_model.documents
        ],
        'vocabulary_sample': vsm_model.vocabulary[:50],
        'vector_sample': {
            'first_document': vsm_model.documents[0]['filename'],
            'vector_length': len(vsm_model.document_vectors[0]),
            'non_zero_entries': sum(1 for w in vsm_model.document_vectors[0] if w > 0),
            'sample_values': [
                {
                    'term': vsm_model.vocabulary[i],
                    'weight': float(vsm_model.document_vectors[0][i])
                }
                for i in range(min(10, vocab_size)) if vsm_model.document_vectors[0][i] > 0
            ]
        }
    }
    
    with open(os.path.join(output_dir, 'vsm_model.json'), 'w', encoding='utf-8') as f:
        json.dump(vsm_data, f, indent=2)
   
    summary = {
        'total_files_found': vsm_model.stats['total_files'],
        'processed_files': vsm_model.stats['processed_files'],
        'failed_files': vsm_model.stats['failed_files'],
        'documents_count': len(vsm_model.documents),
        'vocabulary_size': len(vsm_model.vocabulary),
        'vector_dimensions': vocab_size
    }
    
    return {
        'vsm_model': vsm_model,
        'summary': summary,
        'model_dir': output_dir
    }


def get_user_queries_and_vsm_rank(vsm_model, output_dir):
   
    search_results = []
    os.makedirs(output_dir, exist_ok=True)
    query_count = 0
    
    while True:

        query = input("\nEnter search query (or 'exit' to quit): ").strip()
        
        if query.lower() == 'exit':
            IR_menu()
            break
        
        if not query:
            print("Please enter a query.")
            continue
        
        query_count += 1
        print(f"\nQuery num{query_count}: '{query}'")
        ranked_results = rank_with_vsm_cosine_similarity(vsm_model, query)
        
        if not ranked_results:
            print("No relevant documents found.")
            continue
     
        print(f"\nTOP 10 RANKED RESULTS :")
        print("-" * 90)
        print(f"{'Rank':^6} | {'Similarity':^12} | {'Document':^40} | {'Vector Dim':^10} | {'Words':^8}")
        print("-" * 90)
        
        for rank, result in enumerate(ranked_results[:10], 1):
            filename = result['filename']
            if len(filename) > 38:
                filename = filename[:35] + "..."
            
            print(f"{rank:^6} | {result['similarity']:^12.6f} | {filename:^40} | "
                  f"{result['vector_dim']:^10} | {result['word_count']:^8}")
        
        print("-" * 90)
        
        timestamp = datetime.now().strftime("%H%M%S")
        result_file = os.path.join(output_dir, f"vsm_query_{query_count:03d}.json")
        
        result_data = {
            'query': query,
            'query_number': query_count,
            'timestamp': datetime.now().isoformat(),
            'vector_space_model': 'TF-IDF Weighted Vectors',
            'similarity_metric': 'Cosine Similarity',
            'total_documents': len(vsm_model.documents),
            'relevant_documents_found': len(ranked_results),
            'top_10_results': [
                {
                    'rank': i + 1,
                    'document_id': r['doc_id'],
                    'filename': r['filename'],
                    'cosine_similarity': float(r['similarity']),
                    'matching_terms': r['matching_terms'][:10],
                    'vector_dimensions': r['vector_dim'],
                    'word_count': r['word_count']
                }
                for i, r in enumerate(ranked_results[:10])
            ],
            'query_vector_info': {
                'dimensions': len(ranked_results[0]['query_vector']),
                'non_zero_entries': sum(1 for v in ranked_results[0]['query_vector'] if v > 0)
            }
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2)
        
        search_results.append({
            'query': query,
            'results': ranked_results,
            'file': result_file
        })
        
        print(f"\nSuccessfully VSM results saved named: {os.path.basename(result_file)}")
    
    print(f"\nTotal VSM queries processed: {query_count}")
    return search_results


def rank_with_vsm_cosine_similarity(vsm_model, query):
    query_lower = query.lower()
    query_words = []
    current_word = ""
    
    for char in query_lower:
        if 'a' <= char <= 'z':
            current_word += char
        else:
            if len(current_word) >= 3:
                query_words.append(current_word)
            current_word = ""
    
    if len(current_word) >= 3:
        query_words.append(current_word)
    
    if not query_words:
        return []
    
    print(f"Query: {query_words}")
    
    query_vector = [0.0] * len(vsm_model.vocabulary)
    word_counts = {}
    
    for word in query_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    for word, count in word_counts.items():
        if word in vsm_model.term_index:
            idx = vsm_model.term_index[word]
            if count > 0:
                tf = 1 + math.log(count)
                idf = vsm_model.idf_values.get(word, 0)
                query_vector[idx] = tf * idf
    
   
    results = []
    query_array = np.array(query_vector)
    
    for doc_idx, doc in enumerate(vsm_model.documents):
        doc_vector = vsm_model.document_vectors[doc_idx]
        
        if np.all(query_array == 0):
            continue
        
        doc_array = np.array(doc_vector)
        dot_product = np.dot(query_array, doc_array)
     
        if dot_product == 0:
            continue
 
        query_norm = np.linalg.norm(query_array)
        doc_norm = np.linalg.norm(doc_array)
        
        if query_norm > 0 and doc_norm > 0:
            similarity = dot_product / (query_norm * doc_norm)
            
            if similarity > 0:
                matching_terms = []
                for word in query_words:
                    if word in vsm_model.term_index:
                        idx = vsm_model.term_index[word]
                        if query_vector[idx] > 0 and doc_vector[idx] > 0:
                            matching_terms.append(word)
                
                results.append({
                    'doc_id': doc_idx,
                    'filename': doc['filename'],
                    'relative_path': doc['relative_path'],
                    'similarity': similarity,
                    'word_count': doc['word_count'],
                    'vector_dim': len(doc_vector),
                    'matching_terms': matching_terms,
                    'query_vector': query_vector  
                })
   
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results

def display_vsm_statistics(vsm_model):
    
    print("\n" + "=" * 70)
    print("VECTOR SPACE MODEL STATISTICS")
    print("=" * 70)
    
    print(f"\nModel Information:")
    print(f"  Documents: {len(vsm_model.documents):,}")
    print(f"  Vocabulary: {len(vsm_model.vocabulary):,} terms")
    print(f"  Vector Dimensions: {len(vsm_model.vocabulary)}")
    print(f"\nVector Statistics:")
  
    total_elements = len(vsm_model.document_vectors) * len(vsm_model.vocabulary)
    non_zero_count = 0
    
    for doc_vector in vsm_model.document_vectors:
        non_zero_count += sum(1 for w in doc_vector if w > 0)
    
    sparsity = (total_elements - non_zero_count) / total_elements * 100
    density = 100 - sparsity
    
    print(f"  Total Vector Elements: {total_elements:,}")
    print(f"  Non-zero Elements: {non_zero_count:,}")
    print(f"  Vector Density: {density:.2f}%")
    print(f"  Vector Sparsity: {sparsity:.2f}%")
    
    vector_lengths = []
    for doc_vector in vsm_model.document_vectors:
        magnitude = math.sqrt(sum(w * w for w in doc_vector))
        vector_lengths.append(magnitude)
    
    if vector_lengths:
        avg_length = sum(vector_lengths) / len(vector_lengths)
        print(f"  Average Vector Length: {avg_length:.4f}")
        print(f"  Min Vector Length: {min(vector_lengths):.4f}")
        print(f"  Max Vector Length: {max(vector_lengths):.4f}")

    print(f"\nSample Document Vector (First Document):")
    print(f"  Document: {vsm_model.documents[0]['filename'][:40]}")
    print(f"  Vector Dimensions: {len(vsm_model.document_vectors[0])}")
  
    first_vector = vsm_model.document_vectors[0]
    term_weights = []
    
    for i, weight in enumerate(first_vector):
        if weight > 0:
            term_weights.append((vsm_model.vocabulary[i], weight))
    
    term_weights.sort(key=lambda x: x[1], reverse=True)
    
    print(f"  Top 5 Weighted Terms:")
    for i, (term, weight) in enumerate(term_weights[:5], 1):
        print(f"    {i}. '{term}': {weight:.4f}")


def demonstrate_cosine_similarity_example():

    print("\n" + "=" * 70)
    print("COSINE SIMILARITY DEMONSTRATION")
    print("=" * 70)
    
    print("\nExample: Two documents in a 3-dimensional vector space")
   
    doc1_vector = np.array([1.0, 2.0, 3.0])
    doc2_vector = np.array([4.0, 5.0, 6.0])
    query_vector = np.array([1.0, 1.0, 1.0])
    
    print(f"\nDocument 1 Vector: {doc1_vector}")
    print(f"Document 2 Vector: {doc2_vector}")
    print(f"Query Vector: {query_vector}")
    
    def cos_sim(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    sim1 = cos_sim(query_vector, doc1_vector)
    sim2 = cos_sim(query_vector, doc2_vector)
    
    print(f"\nCosine Similarity Results:")
    print(f"  Query-Doc1 Similarity: {sim1:.4f}")
    print(f"  Query-Doc2 Similarity: {sim2:.4f}")
    
    if sim1 > sim2:
        print(f"→ Document 1 is more similar to the query")
    else:
        print(f"  → Document 2 is more similar to the query")
    
    print("\nFormula: Cosine Similarity = (A·B) / (||A|| × ||B||)")
    print("Where:")
    print("  A·B = Dot product of vectors A and B")
    print("  ||A|| = Euclidean norm (length) of vector A")
    print("  ||B|| = Euclidean norm (length) of vector B")


#--------------------------------------------------------------------- END VSM Section --------------------------------------------------------------

def main():
    menu()
    

if __name__ == "__main__":
    main()
