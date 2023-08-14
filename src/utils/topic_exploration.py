import numpy as np
import pandas as pd


def topic_top_term(model):
    topic_list = list(model.get_topics().keys())
    top_terms_list = []
    for topic_id, term_prob in model.get_topics().items():
        terms = [term for term,prob in term_prob]
        top_terms_list.append( str(topic_id) + '_'+ ' '.join(terms))
    df = pd.DataFrame(list(zip(topic_list, top_terms_list)), columns=['topic_id', 'top_terms'])
    return(df)

def get_top_terms(topic_id):
    top_term_prob_lst = topic_model.get_topic(topic_id)
    top_terms = [term for
                 term,prob in top_term_prob_lst]
    return(' '.join(top_terms))
    
        
def get_top_5_topics(idx, probs):
    ind = np.argpartition(probs[idx], -5)[-5:]
    top_topics = list(ind[np.argsort(probs[idx][ind])[::-1]])
    top_probs = [round(prob, 6) for prob in probs[idx][top_topics]]
    return(top_topics, top_probs)

def get_topic_term_matrix(model, df):
    num_topics = len(model.get_topic_info())-1
    topic_list = list(range(num_topics))
    top_term_list = []
    rep_docs_list = []
    url_list = []
    articles_list = []
    for topic_id in topic_list:    
        top_term_prob_lst = model.get_topic(topic_id)
        top_terms = [term for term,prob in top_term_prob_lst]        
        top_term_list.append(top_terms)
        doc = model.get_representative_docs(topic_id)[0]
        df = df[df.clean_text == doc]
        rep_docs_list.append(doc)    

    df = pd.DataFrame({'topic_id': topic_list, 'top_terms':top_term_list, 'rep_docs': rep_docs_list})
    return(df)


def topic_inference(df, topicpath, model):    
    # make a copy of the dataframe, so source dataframe doesnt get changed
    articles_df= df.copy()    
    
    # convert clean_text series to a list for topic model to do inference
    texts = list(articles_df['clean_text'].values)
    
    # use topic_model to do the topic inference
    topics, probs = model.transform(texts)    

    print('topic inference done')
    # assign the newly predicted topic_id and probability to the dataframe 
    articles_df['topic_id'] = topics
    articles_df['topic_prob'] = probs

    columns = ['id', 'date', 'topic_id', 'topic_prob']
    topic_df = articles_df.loc[:, columns]
#     topic_df['id'] = topic_df['id'].astype('int32')
#     topic_df['topic_id'] = topic_df['topic_id'].astype('int')
    return(topic_df)
