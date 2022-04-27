import re
import spacy
import boto3
import xmltodict, json
import pandas as pd
from tqdm.auto import tqdm
import time
nlp = None

# "^m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$"
# roman_num = re.compile("(ix|iv|v?i{0,3})", flags=re.IGNORECASE)
# roman_num = re.compile("((X[CL]|L?X{0,3})(I[XV]|V?I{0,3}))", flags=re.IGNORECASE)

roman_num = re.compile(
    "(\s|\()(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})(\s|\))",
    flags=re.IGNORECASE
) ## VI
word_period = re.compile('^(\s*\w+\.\s*)', flags=re.IGNORECASE)  ## A.
word_paren = re.compile('(\s*\(\w+\)\s*)', flags=re.IGNORECASE)  ## (A)
num_period = re.compile('^(\s*\d+\.\s*)', flags=re.IGNORECASE)  ## 1.
num_paren = re.compile('(\s*\(\d+\)\s*)', flags=re.IGNORECASE)  ## (1)
ol_breaks = [roman_num, word_period, word_paren, num_period, num_paren]


def remove_ol(text):
    ## remove all initial lists.
    old_len = len(text)
    new_len = 10e8
    while old_len != new_len:
        old_len = new_len
        for ol_break in ol_breaks:
            text = re.sub(ol_break, '', text)
        new_len = len(text)
    return text


def has_subject(text, return_nsubj=False):
    text = remove_ol(text)
    global nlp
    if nlp is None:
        nlp = spacy.load('en_core_web_lg')

    ## search for nsubj
    doc = nlp(text)
    nsubj_toks = [tok for tok in doc if ((tok.dep_ in {"nsubj"}) and (tok.pos_ == 'NOUN'))]
    contains_nsubj = len(nsubj_toks) > 0
    if return_nsubj:
        return contains_nsubj, nsubj_toks
    else:
        return contains_nsubj


def join_paragraphs(paragraphs):
    output_list = []
    prev_par = []
    list_has_nominal = False
    for s in paragraphs:
        if has_subject(s):
            if list_has_nominal:
                output_list.append('\n\n'.join(prev_par))
                prev_par = [s]
            else:
                ## only if the first several paragraph do not have nsubj
                prev_par.append(s)
                list_has_nominal = True
        else:
            prev_par.append(s)
    if prev_par != '':
        output_list.append('\n\n'.join(prev_par))
    return output_list


class MTurkHandler():
    def __init__(self, environment='production'):
        self._mturk_client = {}
        self.client = self.get_client(env=environment)

    def get_client(self, env='production'):
        """
        Get MTurk client.

        parameters:
            * env: `production` or `sandbox`

        """
        if env in self._mturk_client:
            self.client = self._mturk_client[env]
            return self._mturk_client[env]

        ##### mturk client
        region = 'us-east-1'
        environments = {
          "production": {
            "endpoint": "https://mturk-requester.%s.amazonaws.com" % region,
            "preview": "https://www.mturk.com/mturk/preview"
          },
          "sandbox": {
            "endpoint":
                  "https://mturk-requester-sandbox.%s.amazonaws.com" % region,
            "preview": "https://workersandbox.mturk.com/mturk/preview"
          },
        }

        session = boto3.Session(profile_name='mturk')
        self._mturk_client[env] = session.client(
            service_name='mturk',
            region_name=region,
            endpoint_url=environments[env]['endpoint'],
        )
        self.client = self._mturk_client[env]
        return self._mturk_client[env]


    def get_all_reviewable_hits(self):
        next_tok = None
        all_reviewable_hit_ids = []
        reviewable_hits = self.client.list_reviewable_hits()
        next_tok = reviewable_hits['NextToken']
        all_reviewable_hit_ids += reviewable_hits['HITs']
        while 'NextToken' in reviewable_hits:
            next_tok = reviewable_hits['NextToken']
            reviewable_hits = self.client.list_reviewable_hits(NextToken=next_tok)
            all_reviewable_hit_ids += reviewable_hits['HITs']
        all_reviewable_hit_ids = list(map(lambda x: x['HITId'], all_reviewable_hit_ids))
        return all_reviewable_hit_ids


    def get_answers_for_hit_list(self, hit_list):
        """Returns `answers`, a dict of everything Amazon returns, and `answer_df`, a formatted DF for analysis."""
        answers = []
        answer_dfs = []
        for hit_id in tqdm(hit_list):
            ##
            time.sleep(.5)

            assignmentsList = self.client.list_assignments_for_hit(
                HITId=hit_id,
                # AssignmentStatuses=['Submitted', 'Approved'],
                MaxResults=10
            )
            assignments = assignmentsList['Assignments']
            assignments_submitted_count = len(assignments)
            for assignment in assignments:
                # Retreive the attributes for each Assignment
                answer_dict = xmltodict.parse(assignment['Answer'])
                answer = json.loads(answer_dict['QuestionFormAnswers']['Answer'][1]['FreeText'])
                answers.append(answer)
                answer_df = pd.DataFrame(answer['annotated connections'])
                answer_df['worker_id'] = assignment['WorkerId']
                answer_df['assignment_id'] = assignment['AssignmentId']
                answer_df['hit_id'] = assignment['HITId']
                answer_df['time_delta'] = assignment['SubmitTime'] - assignment['AcceptTime']
                answer_dfs.append(answer_df)
        return answers, pd.concat(answer_dfs)

    def reject_assignment(self, assignment_list, comment=None):
        if comment is None:
            comment = 'You didn\'t select anything!'

        failed = 0
        for assignment_id in tqdm(assignment_list):
            try:
                response = self.client.reject_assignment(
                    AssignmentId=assignment_id,
                    RequesterFeedback=comment
                )
            except:
                failed += 1

    def block_workers(self, worker_list, comment=None):
        if comment is None:
            comment = 'You did not fill out any questions on our form.'
        for worker_id in worker_list:
            response = self.client.create_worker_block(
                WorkerId=worker_id,
                Reason=comment
            )

    def bonus_worker(self, worker_id, assignment_id, amount, comment=None):
        if comment is None:
            comment = "For exceptionally detailed work and/or communication with us."

        return self.client.send_bonus(
            WorkerId=worker_id,
            BonusAmount=str(amount),
            AssignmentId=assignment_id,
            Reason=comment,
        )

    def send_message(self, worker_ids, message, subject=None):
        subject = subject or 'A Message Regarding Law-Highlighting Annotations'
        response = self.client.notify_workers(
            Subject=subject,
            MessageText=message,
            WorkerIds=worker_ids
        )

    def create_qualification(self, name):
        response = self.client.create_qualification_type(
            Name=name,
            Keywords='custom-group filtering',
            Description='A custom qualification group given to workers we deem good.',
            QualificationTypeStatus='Active',
        )
        return response


    def give_qualification_to_workers(self, worker_ids, qualification_id):
        """

        """

        for worker_id in worker_ids:
            response = self.client.associate_qualification_with_worker(
                QualificationTypeId=qualification_id,
                WorkerId=worker_id,
                IntegerValue=100,
                SendNotification=True,
            )


def format_html_for_ipython(html):
    """Replace the class names with their colors in CSS."""

    style_dict = {
        'subset-obj':  '" style="background-color: #FDD3C5',
        'subset-qual': '" style="background-color: #FF96BB;',
        'subset-subj': '" style="background-color: #76FF84;',
        'subset-test': '" style="background-color: #2198adb3;',
        'subset-cons': '" style="background-color: #ad21ad6e;',
        'subset-def':  '" style="background-color: #c4bfaf;',
        'subset-term': '" style="background-color: #716e65;',
    }
    for key, val in style_dict.items():
        html = html.replace(key, val)
    return html