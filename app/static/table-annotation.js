let DIVERSITY_HTML = `<td datatype="gender" class="gender my-border-left">
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

let AFFIL_ROLE_HTML = `<td datatype="affiliation" class="affiliation">
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
                                <option value="academic">
                                    Academic
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
                                <option value="unknown" selected="selected">
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

let ERROR_HTML = `<td datatype="error" class="error my-border-right">
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
                                <option value="false_positive_sentence_quote" class="false_positive_sentence_quote">
                                    False positive: We said this was a QUOTE but it doesn't mention any source.
                                </option>
                                <option value="false_positive_sentence_background" class="false_positive_sentence_background">
                                    False positive: We said this was BACKGROUND but it doesn't mention any source.
                                </option>
                                <option disabled role=separator>
                                <option value="mixed_roles_quote" class="mixed_roles_quote">
                                    Type mixed: We tagged this BACKGROUND, but it should be tagged QUOTE.
                                </option>
                                <option value="mixed_roles_background" class="mixed_roles_background">
                                    Type mixed: We tagged this QUOTE, should be tagged BACKGROUND.
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
                                <option value="false_positive_sentence_quote" class="false_positive_sentence_quote">
                                    This shouldn't be tagged as a source sentence. No source mentioned.
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


class TablePageManager {
    constructor(data, task_type) {
        this.source_label_row_num = {}
        this.error_state_dict = {}
        this.source_label_state_dict = {}
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
            that.source_label_state_dict[sent_idx] = 'selected'
            that.source_label_row_num[source_idx] = {
                'selected': sent_idx,
                'vals': source_group['values'].map( function(d){
                    return d.sent_idx
                })
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
        if (error_selected == 'false_negative_wrong_source_new') {
            that.change_source_head_question(row_idx, 'enable')
            that.change_sent_type_question(row_idx, 'enable')
            that.change_source_questions(row_idx, 'enable')
            that.assign_next_source(row_idx, source_idx)
        }
        if (error_selected == 'false_negative_wrong_source_existing') {
            that.change_source_head_question(row_idx, 'enable')
            that.change_sent_type_question(row_idx, 'enable')
            that.change_source_questions(row_idx, 'disable')
        }
        if (error_selected.indexOf('false_positive_sentence') != -1){
            that.clear_source_head(row_idx)
            that.clear_color(row_idx)
            that.assign_next_source(row_idx, source_idx)
        }
        if (error_selected == 'mixed_roles_quote'){
            that.change_source_head_question(row_idx, 'enable')
            that.change_sent_type_question(row_idx, 'enable')
            that.change_sent_type_question_type(row_idx, 'quote')
        }
        if (error_selected == 'mixed_roles_background'){
            that.change_source_head_question(row_idx, 'enable')
            that.change_sent_type_question(row_idx, 'enable')
            that.change_sent_type_question_type(row_idx, 'background')
        }

        // change to no_error
        if (error_selected == 'no_errors') {
            that.reset_input_boxes(row_idx, source_idx)
            that.change_source_head_question(row_idx, 'disable')
            that.change_sent_type_question(row_idx, 'disable')

            let to_disable = true
            // if the old error was some form of false positive
            if (( old_selection.indexOf('false_positive_sentence') != -1 ) | ( old_selection == 'false_negative_wrong_source_new' )) {
                // if the source had previously been completely unselected.
                if (that.source_label_row_num[source_idx]['selected'] == undefined) {
                    that.source_label_row_num[source_idx]['selected'] = parseInt(row_idx)
                    that.source_label_state_dict[row_idx] = 'selected'
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
                that.source_label_state_dict[row_idx] = ''
                that.change_source_questions(row_idx, 'disable')
            }
        }
    }

    unique(list) {
        let result = [];
        $.each(list, function (i, e) {
            if ($.inArray(e, result) == -1) result.push(e);
        });
        return result;
    }

    get_enable_switch(command) {
        if (command == 'enable') {
            return false
        } else {
            return true
        }
    }

    change_affil_questions(row_number, command) {
        let sel = '#row_' + row_number
        let bool = this.get_enable_switch(command)
        $(sel).find('.affiliation').find('select').prop("disabled", bool)
        $(sel).find('.role').find('select').prop("disabled", bool)
        $(sel).find('.role_status').find('select').prop("disabled", bool)
    }

    change_sent_type_question(row_number, command) {
        let sel = '#row_' + row_number
        let bool = this.get_enable_switch(command)
        $(sel).find('.quote_type').find('select').prop("disabled", bool)
    }

    change_sent_type_question_type(row_number, type) {
        let sel = '#row_' + row_number
        $(sel).find('.quote_type').find('select').find('.' + type).prop('selected', true)
    }

    change_source_head_question(row_number, command) {
        let sel = '#row_' + row_number
        let bool = this.get_enable_switch(command)
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
            let bool = this.get_enable_switch(command)
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
        $(sel).find('.error').find('.false_positive_sentence_quote').prop('disabled', true)
        $(sel).find('.error').find('.mixed_roles_background').prop('disabled', true)
        $(sel).find('.error').find('.false_negative_source_named_uncaught').prop('disabled', true)
        $(sel).find('.error').find('.false_negative_source_named_caught').prop('disabled', true)
        $(sel).find('.error').find('.false_negative_source_unnamed_uncaught').prop('disabled', true)
        $(sel).find('.error').find('.false_negative_source_unnamed_existing').prop('disabled', true)
    }

    set_error_source_quote(row_number) {
        let sel = '#row_' + row_number
        $(sel).find('.error').find('.false_positive_sentence_background').prop('disabled', true)
        $(sel).find('.error').find('.mixed_roles_quote').prop('disabled', true)
        $(sel).find('.error').find('.false_negative_source_named_uncaught').prop('disabled', true)
        $(sel).find('.error').find('.false_negative_source_named_caught').prop('disabled', true)
        $(sel).find('.error').find('.false_negative_source_unnamed_uncaught').prop('disabled', true)
        $(sel).find('.error').find('.false_negative_source_unnamed_existing').prop('disabled', true)
    }

    set_error_nothing(row_number) {
        let sel = '#row_' + row_number
        $(sel).find('.error').find('.false_positive_sentence_quote').prop('disabled', true)
        $(sel).find('.error').find('.false_positive_sentence_background').prop('disabled', true)
        $(sel).find('.error').find('.mixed_roles_quote').prop('disabled', true)
        $(sel).find('.error').find('.mixed_roles_background').prop('disabled', true)
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
        let all_possible_rows = this.source_label_row_num[source_idx]['vals']
        return all_possible_rows
            .filter(function (d) {
                return d != curr_row
            })
            .filter(function (d) {
                return that.source_label_state_dict[d] != 'unselected'
            })
    }

    clear_source_head(row_idx) {
        $('#row_' + row_idx).find('.head').find('input').val('')
        $('#row_' + row_idx).find('.na').prop('selected', true)
    }

    assign_next_source(row_idx, source_idx) {
        let active_source_row = parseInt(this.source_label_row_num[source_idx]['selected'])
        let open_source_rows = this.get_possible_source_rows(row_idx, source_idx)
        let to_close;
        let next_row_idx;

        // If the source row is currently actively collecting diversity information, we need to look for another row associated with this source.
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
            this.source_label_state_dict[row_idx] = 'unselected'
            this.source_label_state_dict[next_row_idx] = 'selected'
            this.source_label_row_num[source_idx]['selected'] = next_row_idx
        }
        // otherwise, just mark that this row is unavailable for some future collection effort.
        else {
            this.source_label_state_dict[row_idx] = 'unselected'
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
        return parseInt(d3.max(d3.keys(this.source_label_row_num)))
    }

    register_typed_source(row_idx) {
        let max_source_idx = this.max_source_idx()
        let source_head = $('#row_' + row_idx).find('.head').find('input').val()
        if (this.source_head_to_idx[source_head] === undefined) {
            max_source_idx++
            this.source_head_to_idx[source_head] = max_source_idx
            this.source_label_row_num[max_source_idx] = {
                'selected': row_idx,
                'vals': [row_idx]
            }
            this.source_label_state_dict[row_idx] = 'selected'

            let taken_colors = Object.values(this.source_idx_to_color)
            let avail_colors = this.unique(
                d3.schemeCategory20
                    .concat(d3.schemeCategory20b)
                    .concat(d3.schemeCategory20c)
            ).filter(function (d) {
                return taken_colors.indexOf(d) == -1
            })

            let chosen_color = avail_colors[0]
            this.source_idx_to_color[max_source_idx] = chosen_color
            this.add_color(row_idx, max_source_idx)
        } else {
            let source_idx = this.source_head_to_idx[source_head]
            this.source_label_row_num[source_idx]['vals'].push(row_idx)
            this.add_color(row_idx, source_idx)
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
                    'Typed ' + str_inj + ' source "' + source_head + '" in row ' + (row_idx + 1) + ' DOESN\'T MATCH any existing sources. ' +
                    'Please double check. If NEW source, please choose the appropriate error-type and retry.'
                )
            } else {
                let source_idx = this.source_head_to_idx[source_head]
                this.add_color(row_idx, source_idx)
            }
        } else {
            if (this.source_head_to_idx[source_head] != undefined) {
                alert(
                    'Typed ' + str_inj + ' source "' + source_head + '" in row ' + (row_idx + 1) + ' MATCHES existing sources. ' +
                    'Please double check. If EXISTING source, please choose the appropriate error-type and retry.' + opt_add
                )
            } else {
                // check with the person
                let add_source = confirm(
                    'You entered ' + str_inj + ' source ' + source_head + ' into row ' + (row_idx + 1) + '. ' +
                    'Please confirm that you\'d like to add this source. Press OK to confirm.'
                )
                if (add_source) {
                    this.register_typed_source(row_idx)
                    this.change_source_head_question(row_idx, 'disable')
                }
            }
        }
    }

    input_cell_check(d, source_selector, field_name, other_field_class) {
        let to_continue = true
        let is_disabled = $(d).find('select').prop('disabled')
        let value = $(d).find(':selected').text().trim()
        if (!is_disabled) {
            if (value == '') {
                alert('Make sure the "' + field_name + '" field for row ' + (source_selector + 1) + ' is filled.')
                to_continue = false
            }
            if (value == 'Other') {
                let other_value = $(d).find(other_field_class).val()
                if (other_value == '') {
                    alert('You selected "Other" for the "' + field_name + '" field in row ' + (source_selector + 1) + '. Make sure it is filled.')
                    to_continue = false
                } else {
                    value = 'Other: ' + other_value
                }
            }
        }
        return [to_continue, value]
    }

    text_cell_check(d, source_selector, field_name) {
        let to_continue = true
        let value = $(d).find('input').val()
        let is_disabled = $(d).find('input').prop('disabled')
        if ((!is_disabled) & (value == undefined | value == '')) {
            alert('Make sure the "' + field_name + '" field for row ' + (source_selector + 1) + ' is filled.')
            to_continue = true
        }
        return [to_continue, value]
    }

    process_source_row(source_idx) {
        let that = this
        let to_continue = true
        let to_continue_item, value, is_disabled;
        let source_selector = this.source_label_row_num[source_idx]['selected']
        let output = {}
        output['s_idx'] = source_selector
        let tr = $('#row_' + source_selector)
        let td = $(tr).find('td')
        td.map(function (i, d) {
            let key = $(d).attr('datatype')
            if (key == 'head') {
                [to_continue_item, value] = that.text_cell_check(d, source_selector, 'Source Head')
            }
            if (key == 'quote_type') {
                [to_continue_item, value] = that.input_cell_check(d, source_selector, 'Sentence Type', '.other_sentence_type')
            }
            if (key == 'affiliation') {
                [to_continue_item, value] = that.input_cell_check(d, source_selector, 'Affiliation', '.other_affiliation')
            }
            if (key == 'role') {
                [to_continue_item, value] = that.input_cell_check(d, source_selector, 'Role', '.other_role')
            }
            if (key == 'role_status') {
                [to_continue_item, value] = that.input_cell_check(d, source_selector, 'Role Status', '.other_status')
            }
            if (key == 'race') {
                [to_continue_item, value] = that.input_cell_check(d, source_selector, 'Race', '.other_race')
            }
            if (key == 'gender') {
                [to_continue_item, value] = that.input_cell_check(d, source_selector, 'Gender', '.other_gender')
            }
            if (key == 'age') {
                [to_continue_item, value] = that.input_cell_check(d, source_selector, 'Age')
            }
            if (key == 'education_level') {
                is_disabled = $(d).find('select').prop('disabled')
                value = $(d).find(':selected').map(function (i, d) {
                    return d.value
                })
                if ((!is_disabled) & (value.length == 0)) {
                    alert('Please selected at least one option for "Education Level" for row ' + (source_selector + 1) + '.')
                    to_continue_item = false
                } else {
                    to_continue_item = true
                    value = value.toArray()
                }

            }
            if (key == 'university') {
                [to_continue_item, value] = that.text_cell_check(d, source_selector, 'Educational Institution')
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
        let selected = Object.values(this.source_label_row_num)
        let that = this
        d3.keys(this.source_label_row_num).forEach(function (d) {
            let [to_continue, output] = that.process_source_row(d)
            all_to_continue = (all_to_continue & to_continue)
            all_output.push(output)
        })
        return [all_output, all_to_continue]
    }

    record_data(output, submit_click_event, do_mturk, start_time, output_fname) {
        if (do_mturk) {            // submit mturk
            $('#data').attr('value', JSON.stringify(output))
            $('#submitButton').trigger(submit_click_event.type);
        } else {
            // submit AJAX
            output['start_time'] = start_time
            output['output_fname'] = output_fname
            //
            $.ajax({
                url: "/post_table",
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify(output),
                success: function (result) {
                    if (result === "success") location.href = "/render_table"
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
        Object.entries(this.source_label_row_num)
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
            if (['full', 'affil-role'].indexOf(that.task_type) != -1) {
                table_row += `
                    <tr id="row_` + i + `">
                        <td>` + (i + 1) + `</td>
                `
            }
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
            table_row += `{{ row.type }}
                            </select>
                        </td>`
            table_row += ERROR_HTML
            table_row += AFFIL_ROLE_HTML
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
                { "width": "1%" ,  "targets": 0},  // sentence idx
                { "width": "34%" , "targets": 1},  // sentence
                { "width": "6%" , "targets":  2},   // source head
                { "width": "5%" , "targets":  3},   // sentence-type
                { "width": "5%" , "targets":  4},   // errors
                { "width": "7%" , "targets":  5},   // affiliation
                { "width": "6%" , "targets":  6},   // role
                { "width": "6%" , "targets":  7},   // role status
                { "width": "6%" , "targets":  8},   // gender
                { "width": "6%" , "targets":  9},   // race
                { "width": "6%" , "targets":  10},   // age
                { "width": "6%" , "targets":  11},  // educational level
                { "width": "6%" , "targets":  12},  // institutions
            ],
            fixedHeader: true,
        });
        this.resize_input()
        this.data.forEach(function(d, i) {
            that.disable_errors(d, i)
        })

        // activate all the right source entries
        Object.entries(this.source_label_row_num)
            .forEach(function(d){
                var selected_sent_idx = d[1]['selected']
                that.change_source_questions(selected_sent_idx, 'enable')
            })

        $('.dataTables_empty').parents('tbody').remove()

        return table_sel
    }
}