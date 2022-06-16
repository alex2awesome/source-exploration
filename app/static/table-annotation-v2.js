let DIVERSITY_HTML = `
<td datatype="gender" class="gender my-border-left">
                            <select size="1" disabled>
                                <option value="unknown" selected="selected">
                                </option>
                                <option value="male">
                                    Male
                                </option>
                                <option value="female">
                                    Female
                                </option>
                                <option value="other">
                                    Other
                                </option>
                                <option value="cant_tell">
                                    Cannot Determine
                                </option>
                            </select>
                            <input class="hidden other other_gender" type="text" datatype="other_gender" placeholder="Please specify">
                            <select size="1" class="hidden knowledge_source knowledge_source_gender">
                                <option value="" class="placeholder" selected="selected" disabled>Where did you find this info?</option>
                                <option value="contextual">Contextual clues in the text</option>
                                <option value="looked_online">I needed to look online</option>
                            </select>
                        </td>
                        <td datatype="race" class="race">
                            <select size="1" disabled>
                                <option value="unknown" selected="selected">
                                </option>
                                <option value="white">
                                    White/European
                                </option>
                                <option value="black">
                                    Black
                                </option>
                                <option value="indigenous">
                                    Indigenous
                                </option>
                                <option value="latinx">
                                    Latin/South American
                                </option>
                                <option value="east_asian">
                                    East Asian
                                </option>
                                <option value="south_asian">
                                    South Asian
                                </option>
                                <option value="middle_east">
                                    Middle Eastern
                                </option>
                                <option value="cant_tell">
                                    Cannot Determine
                                </option>
                                <option class="other" value="other">
                                    Other
                                </option>
                            </select>
                            <input class="hidden other other_race" type="text" datatype="other_race" placeholder="Please specify">
                            <select size="1" class="hidden knowledge_source knowledge_source_race">
                                <option value="" class="placeholder" selected="selected" disabled>Where did you find this info?</option>                                
                                <option value="contextual">Contextual clues in the text</option>
                                <option value="looked_online">I needed to look online</option>
                            </select>
                        </td>
                        <td datatype="age" class="age">
                            <select size="1" disabled>
                                <option value="unknown" selected="selected"></option>
                                <option value="0-10">
                                    1-10
                                </option>
                                <option value="10-20">
                                    11-20
                                </option>
                                <option value="20-30">
                                    21-30
                                </option>
                                <option value="30-40">
                                    31-40
                                </option>
                                <option value="40-50">
                                    41-50
                                </option>
                                <option value="50-60">
                                    51-60
                                </option>
                                <option value="60-70">
                                    61-70
                                </option>
                                <option value="70-80">
                                    71-80
                                </option>
                                <option value="gt_80">
                                    >80
                                </option>
                            </select>
                            <select size="1" class="hidden knowledge_source knowledge_source_age">
                                <option value="" class="placeholder" selected="selected" disabled>Where did you find this info?</option>
                                <option value="contextual">Contextual clues in the text</option>
                                <option value="looked_online">I needed to look online</option>
                            </select>
                        </td>
                        <td datatype="education_level" class="education_level">
                            <select class="degree-multiselect" multiple="multiple" disabled>
                                <option value="bachelor">Bachelor (BA, BS, BFA, etc.)</option>
                                <option value="masters">Masters (MA, MS, MBA, etc.)</option>
                                <option value="law">Law (JD)</option>
                                <option value="medical">Medical (MD)</option>
                                <option value="doctoral">PhD (or equivalent)</option>
                                <option value="other">Other</option>
                                <option value="cant_tell">Cannot Determine</option>
                            </select>
                            <input class="hidden other other_education_level" type="text" datatype="other_education_level" placeholder="Please specify">
                            <select size="1" class="hidden knowledge_source knowledge_source_educational_level">
                                <option value="" class="placeholder" selected="selected" disabled>Where did you find this info?</option>
                                <option value="contextual">Contextual clues in the text</option>
                                <option value="looked_online">I needed to look online</option>
                            </select>
                        </td>
                        <td datatype="university" class="university">
                            <input type="text" disabled>
                            <select size="1" class="hidden knowledge_source knowledge_source_university">
                                <option value="" class="placeholder" selected="selected" disabled>Where did you find this info?</option>
                                <option value="contextual">Contextual clues in the text</option>
                                <option value="looked_online">I needed to look online</option>
                            </select>
                        </td>`

let TAGLINE_HTML = `<td datatype="tagline" class="tagline">
    <input class="tagline_text" type="text" datatype="tagline" disabled>
</td>`

let AFFIL_ROLE_HTML = `
    <td datatype="source_type" class="source_type">
                            <select size="1" disabled>
                                <option value="unknown" selected="selected"></option>
                                <option value="named">
                                    Named Individual
                                </option>
                                <option value="unnamed">
                                    Unnamed Individual
                                </option>
                                <option value="named_group">
                                    Named Group
                                </option>
                                <option value="unnamed_group">
                                    Unnamed Group
                                </option>
                                <option value="document">
                                    Report/Document
                                </option>
                                <option value="vote_poll_group">
                                    Vote/Poll
                                </option>  
                                <option value="database">
                                    Database
                                </option>                                                                
                                <option value="other">
                                    Other
                                </option>
                                <option value="cant_tell">
                                    Cannot Determine
                                </option>
                            </select>
                            <input class="hidden other other_source_type" type="text" datatype="other_source_type">
                        </td>
                        <td datatype="affiliation" class="affiliation">
                            <select size="1" disabled>
                                <option value="unknown" selected="selected"></option>
                                <option value="government">
                                    Government
                                </option>
                                <!--                        -->
                                <option value="corporate">
                                    Corporate
                                </option>
                                <option value="ngo">
                                    NGO
                                </option>
                                <option value="industry_group">
                                    Industry Group
                                </option>                                
                                <option value="religious_group">
                                    Religious Group
                                </option>                                
                                <option value="academic">
                                    Academic
                                </option>
                                <option value="media">
                                    Media
                                </option>
                                <option value="political_group">
                                    Political Group
                                </option> 
                                <option value="union">
                                    Union
                                </option>                                
                                <option value="other_group">
                                    Other Group
                                </option>
                                <!--                        -->
                                <option disabled role=separator>
                                <option value="actor">
                                    Actor
                                </option>
                                <option value="witness">
                                    Witness
                                </option>
                                <option value="Victim">
                                    Victim
                                </option>
                                <option value="other">
                                    Other
                                </option>
                                <option value="cant_tell">
                                    Cannot Determine
                                </option>
                              </select>
                            <input class="hidden other other_affiliation" type="text" datatype="other_affiliation">
                        </td>
                        <td datatype="role" class="role">
                            <select size="1" disabled>
                                <option value="unknown" selected="selected"></option>
                                <option value="decision_maker">
                                    Decision Maker
                                </option>
                                <option value="participant">
                                    Participant
                                </option>                                
                                <option value="representative">
                                    Representative
                                </option>
                                <option value="informational">
                                    Informational
                                </option>
                                <option value="other">
                                    Other
                                </option>
                                <option value="cant_tell">
                                    Cannot Determine
                                </option>
                            </select>
                            <input class="hidden other other_role" type="text" datatype="other_role">
                        </td>
                        <td datatype="role_status" class="role_status">
                            <select size="1" disabled>
                                <option value="unknown" selected="selected"></option>
                                <option value="current">
                                    Current
                                </option>
                                <option value="former">
                                    Former
                                </option>
                                <option value="other">
                                    Other
                                </option>
                                <option value="cant_tell">
                                    Cannot Determine
                                </option>
                            </select>
                            <input class="hidden other other_status" type="text" datatype="other_status">
                        </td>`

let affil_field_name_mapper = {
    "quote_type": {
        "QUOTE":"quote",
        "BACKGROUND": "background",
        "": "na",
        "PUBLIC SPEECH, NOT TO JOURNO": "speech",
        "COMMUNICATION, NOT TO JOURNO": "written_comms",
        "PUBLISHED WORK": "published_work",
        "STATEMENT": "statement",
        "LAWSUIT": 'lawsuit',
        "vote_poll": "VOTE/POLL",
        "DOCUMENT": "document",
        "PRESS REPORT": "press_report",
        "TWEET": "tweet",
        "PROPOSAL/ORDER/LAW": "proposal",
        "Data Analysis": "data_analysis",
        "DECLINED COMMENT": "declined_comment",
        "Other": "other"
    },
    "source_type" : {
        "Named Individual": "named",
        "Unnamed Individual": "unnamed",
        "Named Group": "named_group",
        "Unnamed Group": "unnamed_group",
        "Report/Document": "document",
        "Database": "database",
        "Other": "other",
        "Cannot Determine": "cant_tell",
    },
    "affiliation": {
        "": "unknown",
        "Government": "government",
        "Corporate": "corporate",
        "NGO": "ngo",
        "Industry Group": "industry_group",
        "Religious Group": "religious_group",
        "Academic": "academic",
        "Union": "union",
        "Media": "media",
        "Political Group": "political_group",
        "Other Group": "other_group",
        "Actor": "actor",
        "Witness": "witness",
        "Victim": "Victim",
        "Other": "other",
        "Cannot Determine": "cant_tell",
    },
    "role": {
        "": "unknown",
        "Decision Maker": "decision_maker",
        "Participant": "participant",
        "Representative": "representative",
        "Informational": "informational",
        "Other": "other",
        "Cannot Determine": "cant_tell",
    },
    "role_status": {
        "Current": "current",
        "Former": "former",
        "Other": "other",
        "Cannot Determine": "cant_tell",
    },
}

let FULL_ERROR_HTML = `<td datatype="error" class="error my-border-right">
                            <select size="1">
                                <option value="no_errors" selected="selected">
                                    No Errors
                                </option>
                                <option disabled role=separator>
                                <option value="false_negative_source_named_uncaught" class="false_negative_source_named_uncaught">
                                    False negative: This is a named source sentence we failed to label. The source is NOT found elsewhere.
                                </option>
                                <option value="false_negative_source_named_caught" class="false_negative_source_named_caught">
                                    False negative: This is a named source sentence we failed to label. The source IS found elsewhere.
                                </option>
                                <option value="false_negative_source_unnamed_uncaught" class="false_negative_source_unnamed_uncaught">
                                    False negative: This is an organization or an unnamed person we failed to label. The source is NOT found elsewhere.
                                </option>
                                <option value="false_negative_source_unnamed_existing" class="false_negative_source_unnamed_existing">
                                    False negative: This is an organization or an unnamed person we failed to label. The source IS found elsewhere.
                                </option>
                                <!-- -->
                                <option disabled role=separator>
                                <option value="false_positive_sentence" class="false_positive_sentence">
                                    False positive: This sentence does not mention any source.
                                </option>
                                <option disabled role=separator>
                                <option value="wrong_sentence_role" class="wrong_sentence_role">
                                    Type mixed: This sentence is tagged with the wrong role.
                                </option>
                                <option value="false_negative_wrong_source_existing" class="false_negative_wrong_source_existing">
                                    Wrong source: We attributed the quote to the WRONG source. The true source has been found elsewhere.
                                </option>
                                <option value="false_negative_wrong_source_new" class="false_negative_wrong_source_new">
                                    Wrong source: We attributed the quote to the WRONG source. The true source has NOT been found elsewhere.
                                </option>
                                <!--                          -->
                                <option disabled role=separator>
                                <option value="other" class="other">
                                    Other
                                </option>
                            </select>
                            <input class="hidden" type="text" datatype="other_error">
                        </td>`

let ERROR_HTML = `<td datatype="error" class="error my-border-right">
                            <select size="1">
                                <option value="no_errors" selected="selected">
                                    No Errors
                                </option>
                                <option disabled role=separator>
                                <option value="false_negative_source_uncaught" class="false_negative_source_uncaught">
                                    False negative: This is a source sentence we failed to label. The source is NOT found elsewhere.
                                </option>
                                <option value="false_negative_source_caught" class="false_negative_source_caught">
                                    False negative: This is a source sentence we failed to label. The source IS found elsewhere.
                                </option>
                                <!-- -->
                                <option disabled role=separator>
                                <option value="false_positive_sentence" class="false_positive_sentence">
                                    False positive: This sentence does not mention any source.
                                </option>
                                <option disabled role=separator>
                                <option value="wrong_sentence_role" class="wrong_sentence_role">
                                    Type mixed: This sentence is tagged with the wrong role.
                                </option>
                                <option value="false_negative_wrong_source_existing" class="false_negative_wrong_source_existing">
                                    Wrong source: We attributed the quote to the WRONG source. The true source has been found elsewhere.
                                </option>
                                <option value="false_negative_wrong_source_new" class="false_negative_wrong_source_new">
                                    Wrong source: We attributed the quote to the WRONG source. The true source has NOT been found elsewhere.
                                </option>
                                <!--                          -->
                                <option disabled role=separator>
                                <option value="other" class="other">
                                    Other
                                </option>
                            </select>
                            <input class="hidden" type="text" datatype="other_error">
                        </td>`

let DIVERSITY_ERROR_ONLY = `<td datatype="error" class="error my-border-right">
                            <select size="1">
                                <option value="no_errors" selected="selected">
                                    No Errors
                                </option>
                                <option disabled role=separator>
                                <option value="false_positive_sentence" class="false_positive_sentence">
                                    False positive: This sentence does not mention any source.
                                </option>
                                <option value="false_negative_wrong_source_new" class="false_negative_wrong_source_new">
                                    Wrong source: We attributed the quote to the WRONG source.
                                </option>
                                <!--                          -->
                                <option disabled role=separator>
                                <option value="other" class="other">
                                    Other
                                </option>
                            </select>
                            <input class="hidden" type="text" datatype="other_error">
                        </td>`


function unique(list) {
    let result = [];
    $.each(list, function (i, e) {
        if ($.inArray(e, result) == -1) result.push(e);
    });
    return result;
}

function get_enable_switch(command) {
    if (command == 'enable') {
        return false
    } else {
        return true
    }
}

function get_avail_colors(taken_colors){
    let all_colors = d3.schemeCategory20
                    .concat(d3.schemeCategory20b)
                    .concat(d3.schemeCategory20c)
    all_colors = unique(all_colors)
    return all_colors.filter(function (d) { return taken_colors.indexOf(d) == -1 })
}

Array.prototype.remove = function() {
    var what, a = arguments, L = a.length, ax;
    while (L && this.length) {
        what = a[--L];
        while ((ax = this.indexOf(what)) !== -1) {
            this.splice(ax, 1);
        }
    }
    return this;
};

class TablePageManager {
    constructor(data, task_type) {
        this.source_row_nums = {}
        this.error_state_dict = {}
        this.row_source_state_dict = {}
        this.source_head_to_idx = {}
        this.source_idx_to_color = {}
        this.data = data
        this.task_type = task_type
        this.init()
    }

    init(){
        let that = this
        let grouped_keys = d3.nest()
            .key(function (d) { return d.source_idx })
            .sortValues(function(a, b) { return ((a.sent_idx < b.sent_idx) ? -1 : 1); return 0;} )
            .entries(this.data)
            .filter(function(d){ return d['key'] != 'null'})

        // populate source groups
        grouped_keys.forEach(function(source_group){
            let sent_idx = source_group['values'][0]['sent_idx']
            let source_idx = source_group['key']
            let source_head = source_group['values'][0]['head']
            that.source_head_to_idx[source_head] = source_idx
            that.source_idx_to_color[source_idx] = source_group['values'][0]['color']
            let source_sent_idxs = source_group['values'].map( function(d){ return d.sent_idx })
            source_sent_idxs.forEach(function(d) { 
                that.row_source_state_dict[d] = {
                    'status': '',
                    'source_idx': source_idx
                }
            })
            that.row_source_state_dict[sent_idx] = {
                'status': 'selected',
                'source_idx': source_idx
            }
            that.source_row_nums[source_idx] = {
                'selected': sent_idx,
                'vals': source_sent_idxs
            }
        })

        // init errors
        this.data.forEach(function (d, i) {
            that.error_state_dict[i] = 'no_errors'
        })
    }

    handle_error_toggle(this_obj){
        let that = this
        let row_idx = $(this_obj).parents('tr').attr('id').split('_')[1]
        let error_selected = $(this_obj).find("option:selected").attr('value')
        let old_selection = that.error_state_dict[row_idx]
        that.error_state_dict[row_idx] = error_selected
        let source_idx = that.data[row_idx]['source_idx']
        if (that.task_type == 'full') {
            if (error_selected == 'false_negative_source_named_uncaught') {
                that.change_source_questions(row_idx, 'enable') // turn them on
                that.change_source_head_question(row_idx, 'enable')
                that.change_sent_type_question(row_idx, 'enable')
            }
            if (error_selected == 'false_negative_source_named_caught') {
                that.change_source_head_question(row_idx, 'enable')
                that.change_sent_type_question(row_idx, 'enable')
                that.change_source_questions(row_idx, 'disable')
            }
            if (error_selected == 'false_negative_source_unnamed_uncaught') {
                that.change_source_head_question(row_idx, 'enable')
                that.change_sent_type_question(row_idx, 'enable')
                that.change_affil_questions(row_idx, 'enable')
            }
            if (error_selected == 'false_negative_source_unnamed_existing') {
                that.change_source_head_question(row_idx, 'enable')
                that.change_sent_type_question(row_idx, 'enable')
            }
        } else {
            if (error_selected == 'false_negative_source_uncaught') {
                that.change_source_questions(row_idx, 'enable') // turn them on
                that.change_source_head_question(row_idx, 'enable')
                that.change_sent_type_question(row_idx, 'enable')
            }
            if (error_selected == 'false_negative_source_caught') {
                that.change_source_head_question(row_idx, 'enable')
                that.change_sent_type_question(row_idx, 'enable')
                that.change_source_questions(row_idx, 'disable')
            }
        }
        if (error_selected == 'false_negative_wrong_source_new') {
            that.assign_next_source(row_idx, source_idx)
            that.change_source_head_question(row_idx, 'enable')
            that.change_sent_type_question(row_idx, 'enable')
            that.change_source_questions(row_idx, 'enable')
        }
        if (error_selected == 'false_negative_wrong_source_existing') {
            that.assign_next_source(row_idx, source_idx)
            that.change_source_head_question(row_idx, 'enable')
            that.change_sent_type_question(row_idx, 'enable')
            that.change_source_questions(row_idx, 'disable')
        }
        if (error_selected == 'false_positive_sentence'){
            that.clear_source_head(row_idx)
            that.clear_color(row_idx)
            that.assign_next_source(row_idx, source_idx)
        }
        if (error_selected == 'wrong_sentence_role'){
            that.change_source_head_question(row_idx, 'enable')
            that.change_sent_type_question(row_idx, 'enable')
        }
        // if (error_selected == 'mixed_roles_background'){
        //     that.change_source_head_question(row_idx, 'enable')
        //     that.change_sent_type_question(row_idx, 'enable')
        //     that.change_sent_type_question_type(row_idx, 'background')
        // }

        // change to no_error
        if (error_selected == 'no_errors') {
            that.reset_input_boxes(row_idx, source_idx)
            that.change_source_head_question(row_idx, 'disable')
            that.change_sent_type_question(row_idx, 'disable')

            let to_disable = true
            // if the old error was some form of false positive
            let is_false_positive = ( old_selection == 'false_positive_sentence') | ( old_selection == 'false_negative_wrong_source_new' )
            if (is_false_positive) {
                // if the source had previously been completely unselected.
                if (that.source_row_nums[source_idx]['selected'] == undefined) {
                    that.source_row_nums[source_idx]['selected'] = parseInt(row_idx)
                    that.row_source_state_dict[row_idx] = {
                        'status': 'selected',
                        'source_idx': source_idx
                    }
                    that.change_source_questions(row_idx, 'enable')
                    to_disable = false
                }
            }
            if ((old_selection == 'false_negative_wrong_source_new' ) | (old_selection == 'false_negative_source_named_uncaught')){

            }
            if (that.data[row_idx]['color'] === undefined){
                that.clear_color(row_idx)
            }
            if (to_disable) {
                that.row_source_state_dict[row_idx] = {'status': '', 'source_idx': ''}
                that.change_source_questions(row_idx, 'disable')
            }
        }
    }

    change_affil_questions(row_number, command) {
        let sel = '#row_' + row_number
        let bool = get_enable_switch(command)
        $(sel).find('.tagline').find('input').prop("disabled", bool)
        $(sel).find('.source_type').find('select').prop("disabled", bool)
        $(sel).find('.affiliation').find('select').prop("disabled", bool)
        $(sel).find('.role').find('select').prop("disabled", bool)
        $(sel).find('.role_status').find('select').prop("disabled", bool)
    }

    change_sent_type_question(row_number, command) {
        let sel = '#row_' + row_number
        let bool = get_enable_switch(command)
        $(sel).find('.quote_type').find('select').prop("disabled", bool)
    }

    change_sent_type_question_type(row_number, type) {
        let sel = '#row_' + row_number
        $(sel).find('.quote_type').find('select').find('.' + type).prop('selected', true)
    }

    change_source_head_question(row_number, command) {
        let sel = '#row_' + row_number
        let bool = get_enable_switch(command)
        $(sel).find('.head').find('input').prop("disabled", bool)
    }

    change_diversity_questions(row_number, command) {
        let sel = '#row_' + row_number

        if (command == 'remove'){
            $(sel).find('.gender').find('select').addClass('hidden')
            $(sel).find('.race').find('select').addClass('hidden')
            $(sel).find('.age').find('select').addClass('hidden')
            $(sel).find('.education_level').find('select').addClass('hidden')
            $(sel).find('.university').find('input').addClass('hidden')
            $(sel).find('.knowledge_source').find('select').addClass('hidden')
        } else {
            let bool = get_enable_switch(command)
            $(sel).find('.gender').find('select').removeClass('hidden').prop("disabled", bool)
            $(sel).find('.race').find('select').removeClass('hidden').prop("disabled", bool)
            $(sel).find('.age').find('select').removeClass('hidden').prop("disabled", bool)
            if (bool) {
                $(sel).find('.education_level').find('.degree-multiselect').removeClass('hidden').multiselect("disable")
            } else {
                $(sel).find('.education_level').find('.degree-multiselect').removeClass('hidden').multiselect("enable")
            }

            $(sel).find('.university').find('input').removeClass('hidden').prop("disabled", bool)
            $(sel).find('.knowledge_source').find('select').removeClass('hidden').prop("disabled", bool)
        }
    }

    change_source_questions(row_number, command) {
        this.change_affil_questions(row_number, command)
        this.change_diversity_questions(row_number, command)
    }

    change_all_questions(row_number, command) {
        this.change_source_head_question(row_number, command)
        this.change_sent_type_question(row_number, command)
        this.change_affil_questions(row_number, command)
        this.change_source_questions(row_number, command)
    }

    set_error_source_background(row_number) {
        let sel = '#row_' + row_number
        if (this.task_type == 'full') {
            $(sel).find('.error').find('.false_negative_source_named_uncaught').prop('disabled', true)
            $(sel).find('.error').find('.false_negative_source_named_caught').prop('disabled', true)
            $(sel).find('.error').find('.false_negative_source_unnamed_uncaught').prop('disabled', true)
            $(sel).find('.error').find('.false_negative_source_unnamed_existing').prop('disabled', true)
        } else {
            $(sel).find('.error').find('.false_negative_source_uncaught').prop('disabled', true)
            $(sel).find('.error').find('.false_negative_source_caught').prop('disabled', true)
        }
    }

    set_error_source_quote(row_number) {
        let sel = '#row_' + row_number
        if (this.task_type == 'full') {
            $(sel).find('.error').find('.false_negative_source_named_uncaught').prop('disabled', true)
            $(sel).find('.error').find('.false_negative_source_named_caught').prop('disabled', true)
            $(sel).find('.error').find('.false_negative_source_unnamed_uncaught').prop('disabled', true)
            $(sel).find('.error').find('.false_negative_source_unnamed_existing').prop('disabled', true)
        } else {
            $(sel).find('.error').find('.false_negative_source_uncaught').prop('disabled', true)
            $(sel).find('.error').find('.false_negative_source_caught').prop('disabled', true)
        }
    }

    set_error_nothing(row_number) {
        let sel = '#row_' + row_number
        $(sel).find('.error').find('.false_positive_sentence').prop('disabled', true)
        $(sel).find('.error').find('.wrong_sentence_role').prop('disabled', true)
        $(sel).find('.error').find('.false_negative_wrong_source_existing').prop('disabled', true)
        $(sel).find('.error').find('.false_negative_wrong_source_new').prop('disabled', true)
    }

    disable_errors(datum, idx) {
        if (datum['source_idx'] == 'null') {
            this.set_error_nothing(idx)
        } else if (datum['type'] == 'QUOTE') {
            this.set_error_source_quote(idx)
        } else {
            this.set_error_source_background(idx)
        }
    }

    catalogue_typed_source_head_timer(row_idx) {
        let typingTimer;                //timer identifier
        let doneTypingInterval = 2000;  //time in ms, 5 seconds for example
        let input = $('#row_' + row_idx).find('.head').find('input');

        //on keyup, start the countdown
        input.on('keyup', function () {
            clearTimeout(typingTimer);
            typingTimer = setTimeout(doneTyping, doneTypingInterval);
        });

        //on keydown, clear the countdown
        input.on('keydown', function () {
            clearTimeout(typingTimer);
        });
    }

    resize_input() {
        $('input').css('width', function (i, d) {
            return $(this).parent().width()
        })
        $('select').css('width', function (i, d) {
            return $(this).parent().width()
        })
    }

    handle_other_box(that) {
        let multi_bool = (that.attr('class') == 'degree-multiselect')
        let toggle_on
        let t = that.find("option:selected")
        let o = that.parents('td').find('.other')
        if (multi_bool) {
            let selected = t.map(function (i, d) {
                return $(d).attr('value')
            })
            toggle_on = (selected.filter(function (i, d) {
                return d == 'other'
            }).length > 0)
        } else {
            toggle_on = (t.val().indexOf('other') != -1)
        }
        if (toggle_on) {
            o.removeClass('hidden')
        } else {
            o.addClass('hidden')
        }
    }

    get_possible_source_rows(curr_row, source_idx) {
        let that = this
        let all_possible_rows = this.source_row_nums[source_idx]['vals']
        return all_possible_rows
            .filter(function (d) { return d != curr_row })
            .filter(function (d) { return that.row_source_state_dict[d]['status'] != 'unselected' })
    }

    clear_source_head(row_idx) {
        $('#row_' + row_idx).find('.head').find('input').val('')
        $('#row_' + row_idx).find('.na').prop('selected', true)
    }

    assign_next_source(row_idx, source_idx) {
        let active_source_row = parseInt(this.source_row_nums[source_idx]['selected'])
        let open_source_rows = this.get_possible_source_rows(row_idx, source_idx)
        let to_close;
        let next_row_idx;

        // If the source row is currently actively collecting information,
        // we need to look for another row associated with this source.
        if (active_source_row == row_idx) {
            // if there aren't other source rows, check to make sure the user wants to eliminate the source.
            if (open_source_rows.length == 0) {
                to_close = confirm(
                    'This is the last entry we have for this source. ' +
                    'If this is a false positive, it means the source does not exist. ' +
                    'Click OK to confirm.'
                )
                next_row_idx = undefined
            } else {
                to_close = true
                next_row_idx = open_source_rows[0]
            }
        }

        // if we are closing out this row...
        if (to_close) {
            // 1. disable current source questions
            this.change_source_head_question(row_idx, 'disable')
            this.change_sent_type_question(row_idx, 'disable')
            this.change_source_questions(row_idx, 'disable')

            // 2. turn on the next source questions
            this.change_source_questions(next_row_idx, 'enable')
            this.row_source_state_dict[row_idx]['status'] = 'unselected'
            if (next_row_idx != undefined) {
                this.row_source_state_dict[next_row_idx]['status'] = 'selected'
            }
            this.source_row_nums[source_idx]['selected'] = next_row_idx
        }
        // otherwise, just mark that this row is unavailable for some future collection effort.
        else {
            this.row_source_state_dict[row_idx]['status'] = 'unselected'
        }
    }

    clear_color(row_idx) {
        $('#row_' + row_idx).find('.sentence').removeAttr('style')
    }

    add_color(row_idx, source_idx) {
        let color = this.source_idx_to_color[source_idx]
        $('#row_' + row_idx).find('.sentence').css('background-color', color)
    }

    reset_input_boxes(row_idx, source_idx) {
        let row = this.data[row_idx]
        $('#row_' + row_idx).find('.head').find('input').val(row['head'])
        if (row['type'] == "QUOTE") {
            $('#row_' + row_idx).find('.quote').prop('selected', true)
        } else if (row['type'] == "BACKGROUND") {
            $('#row_' + row_idx).find('.background').prop('selected', true)
        } else {
            $('#row_' + row_idx).find('.na').prop('selected', true)
        }
        this.add_color(row_idx, source_idx)
    }

    max_source_idx() {
        return parseInt(d3.max(d3.keys(this.source_row_nums)))
    }

    get_source_color(source_idx){
        let taken_colors = Object.values(this.source_idx_to_color)
        let avail_colors = get_avail_colors(taken_colors)
        return avail_colors[0]
    }

    register_typed_source(row_idx) {
        let source_head = $('#row_' + row_idx).find('.head').find('input').val()
        let source_idx;

        // if we're changing sources, remove row from the old source
        if (this.row_source_state_dict[row_idx] != undefined){
            if (this.row_source_state_dict[row_idx]['source_idx'] != '') {
                let old_source_idx = this.row_source_state_dict[row_idx]['source_idx']
                this.source_row_nums[old_source_idx]['vals'].remove(row_idx)
            }
        }

        // if it's a new source, assign it a new id and generate new colors
        if (this.source_head_to_idx[source_head] === undefined) {
            let max_source_idx = this.max_source_idx()
            source_idx = max_source_idx + 1
            this.source_head_to_idx[source_head] = source_idx
            this.source_row_nums[source_idx] = { 'selected': row_idx, 'vals': [] }
            this.source_idx_to_color[source_idx] = this.get_source_color(source_idx)
        // otherwise, get source_idx from the dicts
        } else {
            source_idx = this.source_head_to_idx[source_head]
        }

        // update state dicts
        this.add_color(row_idx, source_idx)
        this.source_row_nums[source_idx]['vals'].push(row_idx)
        this.row_source_state_dict[row_idx] = {
            'source_idx': source_idx,
            'status': 'selected'
        }
    }

    check_typed_source(row_idx, final_check, unnamed) {
        if (final_check == undefined) {
            final_check = false
        }
        // unnamed handling
        if ((unnamed == undefined) | (unnamed == false)){
            var str_inj = 'NAMED'
            var opt_add = ' If more than one unnamed sources share the same identifier, please differentiate them with numbers (e.g. "anonymous official 1", "anonymous official 2", etc.).'
        } else {
            var str_inj = 'UNNAMED'
            var opt_add = ''
        }

        let source_head = $('#row_' + row_idx).find('.head').find('input').val()
        if (final_check) {
            if (this.source_head_to_idx[source_head] === undefined) {
                alert(
                    'You typed source "' + source_head + '" in row ' + (row_idx + 1) + ', which DOESN\'T MATCH any existing sources. ' +
                    'Please double check. If NEW source, please choose the appropriate error-type and retry.'
                )
            } else {
                let source_idx = this.source_head_to_idx[source_head]
                this.add_color(row_idx, source_idx)
                this.register_typed_source(row_idx)
            }
        } else {
            if (this.source_head_to_idx[source_head] != undefined) {
                alert(
                    'You typed source "' + source_head + '" in row ' + (row_idx + 1) + ', which MATCHES existing sources. ' +
                    'Please double check. If EXISTING source, please choose the appropriate error-type and retry.' + opt_add
                )
            } else {
                // check with the person
                let add_source = confirm(
                    'You entered source ' + source_head + ' into row ' + (row_idx + 1) + '. ' +
                    'Please confirm that you\'d like to add this source. Press OK to confirm.'
                )
                if (add_source) {
                    this.register_typed_source(row_idx)
                    this.change_source_head_question(row_idx, 'disable')
                }
            }
        }
    }

    knowledge_cell_check(d, row_selector, field_name){
        let to_continue = true
        let k_output = null
        let k = $(d).find('.knowledge_source')
        if ($(k).length > 0){
            let k_val = $(k).find(':selected:not(.placeholder)')
            if ($(k_val).length == 0){
                to_continue = false
                if (this.task_type != 'diversity') {
                    alert('Please tell us where you found the information for the "' + field_name + '" field in row ' + (row_selector + 1) + ', using the dropdown below it.')
                } else {
                    alert('Please tell us where you found the information for the "' + field_name + '" field, using the dropdown below it.')
                }
            } else {
                k_output = $(k_val).val()
            }
        }
        return [to_continue, k_output]
    }

    input_cell_check(d, row_idx, field_name, other_field_class, is_source_head_row) {
        let to_continue = true
        let d_sel = $(d).find('select:not(.knowledge_source)')
        // let is_disabled = $(d_sel).prop('disabled') |
        let is_disabled = $(d_sel).hasClass('hidden')
        let value = null
        if (!is_disabled) {
            let field_value = $(d_sel).find(':selected').text().trim()
            if ((field_value == '') & (is_source_head_row)) {
                if (this.task_type != 'diversity') {
                    alert('Make sure the "' + field_name + '" field in row ' + (row_idx + 1) + ' is filled.')
                } else {
                    alert('Make sure the "' + field_name + '" field is filled.')
                }
                to_continue = false
            }
            if (field_value == 'Other') {
                let other_value = $(d).find(other_field_class).val()
                if (other_value == '') {
                    if (this.task_type != 'diversity') {
                        alert('You selected "Other" for the "' + field_name + '" field in row ' + (row_idx + 1) + '. Make sure the "Other" text box is filled.')
                    } else {
                        alert('You selected "Other" for the "' + field_name + '" field. Make sure the "Other" text box is filled.')
                    }
                    to_continue = false
                } else {
                    field_value = 'Other: ' + other_value
                }
            }
            let [to_continue_k, k_value] = this.knowledge_cell_check(d, row_idx, field_name)
            to_continue = (to_continue & to_continue_k)
            value = {
                'field_value': field_value,
                'knowledge_source': k_value,
            }
        }
        return [to_continue, value]
    }

    text_cell_check(d, row_idx, field_name, is_source_head_row) {
        let to_continue = true
        let value = null
        // let is_disabled = $(d).find('input').prop('disabled')
        let is_disabled = $(d).hasClass('hidden')
        if (!is_disabled) {
            let field_value = $(d).find('input').val()
            if ((field_value == undefined | field_value == '') & is_source_head_row) {
                alert('Make sure the "' + field_name + '" field in row ' + (row_idx + 1) + ' is filled.')
                to_continue = false
            }
            let [to_continue_k, k_value] = this.knowledge_cell_check(d, row_idx, field_name)
            to_continue = (to_continue & to_continue_k)
            value = {
                'field_value': field_value,
                'knowledge_source': k_value,
            }
        }
        return [to_continue, value]
    }

    multi_select_check(d, row_idx, field_name, is_source_head_row){
        let to_continue = true
        let value = null
        let d_sel = $(d).find('select:not(.knowledge_source)')
        let is_disabled = ($(d_sel).prop('disabled') | $(d_sel).hasClass('hidden'))
        if (!is_disabled) {
            let field_value = $(d_sel).find(':selected').map(function (i, d) { return d.value })
            if ((field_value.length == 0) & is_source_head_row) {
                if (this.task_type != 'diversity') {
                    alert('Please select at least one option for the "' + field_name + '" field in row ' + (row_idx + 1) + '.')
                } else {
                    alert('Please select at least one option for the "' + field_name + '" field.')
                }
                to_continue = false
            } else {
                field_value = field_value.toArray()
                let [to_continue_k, k_value] = this.knowledge_cell_check(d, row_idx, field_name)
                to_continue = (to_continue & to_continue_k)
                value = {
                    'field_value': field_value,
                    'knowledge_source': k_value,
                }
            }
        }
        return [to_continue, value]
    }

    process_row(row_idx) {
        let that = this
        let to_continue = true
        let to_continue_item, value;
        let is_source_head_row = false
        let output = {}
        output['row_idx'] = row_idx
        if (this.row_source_state_dict[row_idx] != undefined) {
            if (this.row_source_state_dict[row_idx]['source_idx'] != '') {
                let source_idx = this.row_source_state_dict[row_idx]['source_idx']
                let source_head_row = this.source_row_nums[source_idx]['selected']
                output['s_idx'] = source_idx
                is_source_head_row = (source_head_row == row_idx)
            }
        }
        let tr = $('#row_' + row_idx)
        let td = $(tr).find('td')
        td.map(function (i, d) {
            let key = $(d).attr('datatype')
            if (key == 'head') {
                [to_continue_item, value] = that.text_cell_check(d, row_idx, 'Source Head', is_source_head_row)
            }
            if (key == 'error') {
                [to_continue_item, value] = that.input_cell_check(d, row_idx, 'Error Type')
            }
            if (key == 'quote_type') {
                [to_continue_item, value] = that.input_cell_check(d, row_idx, 'Sentence Type', '.other_sentence_type', is_source_head_row)
            }
            if (key == 'source_type') {
                [to_continue_item, value] = that.input_cell_check(d, row_idx, 'Source Type', '.other_source_type', is_source_head_row)
            }
            if (key == 'tagline') {
                [to_continue_item, value] = that.text_cell_check(d, row_idx, 'Tag Line', is_source_head_row)
            }
            if (key == 'affiliation') {
                [to_continue_item, value] = that.input_cell_check(d, row_idx, 'Affiliation', '.other_affiliation', is_source_head_row)
            }
            if (key == 'role') {
                [to_continue_item, value] = that.input_cell_check(d, row_idx, 'Role', '.other_role', is_source_head_row)
            }
            if (key == 'role_status') {
                [to_continue_item, value] = that.input_cell_check(d, row_idx, 'Role Status', '.other_status', is_source_head_row)
            }
            if (key == 'race') {
                [to_continue_item, value] = that.input_cell_check(d, row_idx, 'Race', '.other_race', is_source_head_row)
            }
            if (key == 'gender') {
                [to_continue_item, value] = that.input_cell_check(d, row_idx, 'Gender', '.other_gender', is_source_head_row)
            }
            if (key == 'age') {
                [to_continue_item, value] = that.input_cell_check(d, row_idx, 'Age', is_source_head_row)
            }
            if (key == 'education_level') {
                [to_continue_item, value] = that.multi_select_check(d, row_idx, 'Education Level', is_source_head_row)
            }
            if (key == 'university') {
                [to_continue_item, value] = that.text_cell_check(d, row_idx, 'Educational Institution', is_source_head_row)
            }

            if ((key != undefined) & (key != 'sentence') & (key != 'error')) {
                output[key] = value
                to_continue = (to_continue & to_continue_item)
            }
        })
        return [to_continue, output]
    }

    process_output(output) {
        let all_output = []
        let all_to_continue = true
        let that = this
        this.data.map(function(d){return d.sent_idx}).forEach(function (d) {
            let [to_continue, output] = that.process_row(d)
            all_to_continue = (all_to_continue & to_continue)
            all_output.push(output)
        })
        let data = {
            'row_data': all_output,
            'source_dict': this.source_row_nums,
            'source_heads': this.source_head_to_idx,
            'error_dict': this.error_state_dict
        }

        return [data, all_to_continue]
    }

    record_data(output, submit_click_event, do_mturk, start_time, output_fname, loc) {
        if (do_mturk) {            // submit mturk
            $('#data').attr('value', JSON.stringify(output))
            $('#submitButton').trigger(submit_click_event.type);
        } else {
            // submit AJAX
            output['start_time'] = start_time
            output['output_fname'] = output_fname
            if (output_fname.indexOf('affil-role') != -1){
                loc = loc + "?task=affil-role"
            } else if (output_fname.indexOf('diversity') != -1){
                loc = loc + "?task=diversity"
            }
            //
            $.ajax({
                url: "/post_table",
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify(output),
                success: function (result) {
                    if (result === "success") location.href = loc
                }
            })
        }
    }

    build_table_diversity(table_sel, data) {
        // table body
        let that = this
        let table_body_html = ""
        data = data.filter(function(e, i) {
            return e['source_idx'] != 'null'
        })
        data.sort(function(l, r){
            if  (l['source_idx'] < r['source_idx']) return -1
            else if  (l['source_idx'] > r['source_idx']) return 1
            else {
                if (l['sent_idx'] < r['sent_idx']) return -1
                else return 1
            }

        })

        data.forEach(function (row, i) {
            let table_row = '<tr id="row_' + row['sent_idx'] + '">'

            // sentence
            if (row.color) {
                table_row += `<td datatype='sentence' class="sentence" style="background-color: ` + row.color + `">` + row.sent + `</td>`
            } else {
                table_row += `<td datatype='sentence' class="sentence">` + row.sent + `</td>`
            }

            // source head
            table_row += `<td datatype="head" class="head my-border-left">
                            <input type="text" value="` + row.head + `" disabled>
                          </td>`
            table_row += DIVERSITY_HTML
            table_row += "</tr>"
            table_body_html += table_row
        })

        // append it all to the selector
        table_sel.append(table_body_html)
        table_sel.DataTable({
            paging: false,
            ordering: false,
            searching: false,
            info: false,
            columnDefs: [
                { "width": "34%" , "targets": 0},   // sentence
                { "width": "6%" , "targets":  1},   // source head
                { "width": "6%" , "targets":  2},   // gender
                { "width": "6%" , "targets":  3},   // race
                { "width": "6%" , "targets":  4},   // age
                { "width": "6%" , "targets":  5},   // educational level
                { "width": "6%" , "targets":  6},   // institutions
            ],
            fixedHeader: true,
        });
        this.resize_input()

        // activate all the right source entries
        Object.entries(this.source_row_nums)
            .forEach(function(d){
                var selected_sent_idx = d[1]['selected']
                var not_selected_idxs = d[1]['vals'].filter(function(e){return e != d[1]['selected']})
                // activate one
                that.change_diversity_questions(selected_sent_idx, 'enable')

                $('#row_' + selected_sent_idx).find('.knowledge_source').removeClass('hidden')

                // remove the others
                not_selected_idxs.forEach(function (e) {
                    that.change_diversity_questions(e, 'remove')
                })
            })

        $('.dataTables_empty').parents('tbody').remove()

        return table_sel
    }

    set_select_field(row_idx, data_row, cell_selector, change_value){
        if (data_row[cell_selector] != undefined){
            let stored_value = data_row[cell_selector]
            if (stored_value != null) {
                if (typeof (stored_value) == 'object'){
                    stored_value = stored_value['field_value']
                }
                let select_val = affil_field_name_mapper[cell_selector][stored_value]
                $('#row_' + row_idx).find('.' + cell_selector).find('select').val(select_val)
            }
        }
    }

    set_input_text_field(row_idx, data_row, cell_selector){
        if (data_row[cell_selector] != undefined){
            let stored_value = data_row[cell_selector]
            if (stored_value != null) {
                if (typeof (stored_value) == 'object'){
                    stored_value = stored_value['field_value']
                }
                $('#row_' + row_idx).find('.' + cell_selector).find('input').val(stored_value)
            }
        }
    }

    set_background_color_white(row_idx){
        $('#row_' + row_idx).find('.sentence').css('background-color', 'white')
    }

    determine_error_type(input_row, annotated_row){
        let s_idx = annotated_row['s_idx']
        if ((input_row['head'] == undefined) & (annotated_row['head']['field_value'] == '')) {
            return "no_error"
        }

        else if ((input_row['head'] == undefined) & (annotated_row['head']['field_value'] != '')){
            if (this.source_row_nums[s_idx]['selected'] == annotated_row['row_idx'])
                return "false_negative_source_uncaught"
            else
                return "false_negative_source_caught"
        }

        else if ((input_row['head'] != undefined) & (annotated_row['head']['field_value'] == '')){
            return 'false_positive_sentence'
        }

        else if ((input_row['head'] != annotated_row['head']['field_value'])){
            if (this.source_row_nums[s_idx]['selected'] == annotated_row['row_idx'])
                return "false_negative_wrong_source_new"
            else
                return "false_negative_wrong_source_existing"
        }

        else if (input_row['type'] != annotated_row['quote_type']['field_value'])
            return "wrong_sentence_role"

        else {
            return "no_error"
        }
    }

    populate_data_to_check(annotated_data){
        let that = this
        annotated_data.forEach(function(row, i ){
            let row_idx = row['row_idx']
            that.set_background_color_white(row_idx)
            let s_idx = row['s_idx']
            if (s_idx != undefined) {
                that.change_all_questions(row_idx, 'enable')
                // select fields
                that.set_select_field(row_idx, row, 'quote_type',)
                that.set_select_field(row_idx, row, 'source_type',)
                that.set_select_field(row_idx, row, 'affiliation',)
                that.set_select_field(row_idx, row, 'role',)
                that.set_select_field(row_idx, row, 'role_status')
                // input text fields
                that.set_input_text_field(row_idx, row, 'tagline')
                that.set_input_text_field(row_idx, row, 'head')
                // coloring
                that.register_typed_source(row_idx)
            }
            // error
            let error_type = that.determine_error_type(that.data[i], row)
            $('#row_' + row_idx).find('.error').find('select').val(error_type)
        })

    }

    build_table(table_sel, data, filter_rows) {
        // table body
        let table_body_html = ""
        if (filter_rows){
            data = data.filter(function(e, i) {
                return e['source_idx'] != 'null'
            })
        }
        let that = this
        data.forEach(function (row, i) {
            let table_row = ''
            table_row += `<tr id="row_` + i + `">
                            <td>` + (i + 1) + `</td>`
            // sentence
            if (row.color) {
                table_row += `<td datatype='sentence' class="sentence" style="background-color: ` + row.color + `">` + row.sent + `</td>`
            } else {
                table_row += `<td datatype='sentence' class="sentence">` + row.sent + `</td>`
            }

            // source head
            table_row += `<td datatype="head" class="head my-border-left">`

            if (row.head) {
                table_row += `<input type="text" value="` + row.head + `" disabled>`
            } else {
                table_row += `<input type="text" value="" disabled>`
            }
            table_row += `</td><td datatype="quote_type" class="quote_type">
                <select type="text" disabled>`

            // source quote
            if (row.type == "QUOTE") {
                table_row += `<option value="quote" class="quote" selected="selected">QUOTE</option>`
            } else {
                table_row += `<option value="quote" class="quote">QUOTE</option>`
            }
            if (row.type == "BACKGROUND") {
                table_row += `<option value="background" class="background" selected="selected">BACKGROUND</option>`
            } else {
                table_row += `<option value="background" class="background">BACKGROUND</option>`
            }
            if (row.type == "") {
                table_row += `<option value="na" class="na" selected="selected"></option>`
            } else {
                table_row += `<option value="na" class="na"></option>`
            }
            table_row += `<option value="speech">PUBLIC SPEECH, NOT TO JOURNO</option>
                          <option value="written_comms">COMMUNICATION, NOT TO JOURNO</option>
                          <option value="published_work">PUBLISHED WORK</option>
                          <option value="statement">STATEMENT</option>
                          <option value="lawsuit">LAWSUIT</option>
                          <option value="vote_poll">VOTE/POLL</option>
                          <option value="document">DOCUMENT</option>
                          <option value="press_report">PRESS REPORT</option>
                          <option value="tweet">TWEET</option>
                          <option value="proposal">PROPOSAL/ORDER/LAW</option>
                          <option value="declined_comment">DECLINED COMMENT</option>
                          <option value="other">Other</option>
                    </select>
                    <input class="hidden other other_sentence_type" type="text" datatype="other_sentence_type" placeholder="Please specify">
                </td>`

            table_row += ERROR_HTML
            if (that.task_type == 'affil-role'){
                table_row += TAGLINE_HTML
            }
            table_row += AFFIL_ROLE_HTML
            if (that.task_type == 'full') {
                table_row += DIVERSITY_HTML
            }
            table_row += "</tr>"
            table_body_html += table_row
        })

        // append it all to the selector
        table_sel.append(table_body_html)
        let table_col_defs = [
                { "width": "1%" ,  "targets": 0},  // sentence idx
                { "width": "34%" , "targets": 1},  // sentence
                { "width": "6%" , "targets":  2},   // source head
                { "width": "5%" , "targets":  3},   // sentence-type
                { "width": "5%" , "targets":  4},   // errors
                { "width": "7%" , "targets":  5},   // source-type
                { "width": "7%" , "targets":  6},   // affiliation
                { "width": "6%" , "targets":  7},   // role
                { "width": "6%" , "targets":  8},   // role status
        ]
        let addit_affil_role_col_defs = [
                { "width": "6%" , "targets":  9},   // tagline
        ]
        let addit_diversity_col_defs = [
                { "width": "6%" , "targets":  9},   // gender
                { "width": "6%" , "targets":  10},  // race
                { "width": "6%" , "targets":  11},  // age
                { "width": "6%" , "targets":  12},  // educational level
                { "width": "6%" , "targets":  13},  // institutions
        ]

        if (that.task_type == 'full'){
            table_col_defs = table_col_defs.concat(addit_diversity_col_defs)
        }
        if (that.task_type == 'affil-role'){
            table_col_defs = table_col_defs.concat(addit_affil_role_col_defs)
        }

        table_sel.DataTable({
            paging: false,
            ordering: false,
            searching: false,
            info: false,
            columnDefs: table_col_defs,
            fixedHeader: true,
        });
        this.resize_input()
        this.data.forEach(function(d, i) {
            that.disable_errors(d, i)
        })

        // activate all the right source entries
        Object.entries(this.source_row_nums)
            .forEach(function(d){
                var selected_sent_idx = d[1]['selected']
                that.change_source_questions(selected_sent_idx, 'enable')
            })

        $('.dataTables_empty').parents('tbody').remove()

        return table_sel
    }
}