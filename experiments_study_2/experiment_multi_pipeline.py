import argparse
import pickle
import pandas as pd
from functools import reduce
import sys

from experiment_multi_util import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../Experiment_Effectiveness/classifier_baseline", required=False)
    parser.add_argument("--data_dir", type=str, default='../Experiment_Effectiveness/feedback-prize-effectiveness/train', required=False)
    parser.add_argument("--model", type=str, default="allenai/longformer-large-4096", required=False)
    parser.add_argument("--lr", type=float, default=1e-5, required=False)
    parser.add_argument("--max_len", type=int, default=1024, required=False)
    parser.add_argument("--batch_size", type=int, default=1, required=False)
    parser.add_argument("--epochs", type=int, default=1, required=False)
    parser.add_argument("--max_norm", type=int, default=10, required=False)
    parser.add_argument("--setting_name", type=str, default="test", required=False)
    return parser.parse_args()


# TASK A: Argument type prediction
# TASK B: Argument Quality prediction
if __name__ == "__main__":

    # Step 1. Get args, seed everything and choose device
    args = parse_args()
    seed_everything(42)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    # train_df = pd.read_csv('/'.join([args.input_dir, 'train.csv']))
    # validate_df = pd.read_csv('/'.join([args.input_dir, 'validate.csv']))
    # test_df = pd.read_csv('/'.join([args.input_dir, 'test.csv']))

    train_df = pd.read_csv('/'.join([args.input_dir, 'train.csv']))
    validate_df = pd.read_csv('/'.join([args.input_dir, 'validate.csv']))
    test_df = pd.read_csv('/'.join([args.input_dir, 'test.csv']))

    # train_df = train_df.head(10)
    # validate_df = validate_df.head(10)
    # test_df = test_df.head(10)

    # print(train_df['discourse_effectiveness'])
    # print(validate_df['discourse_effectiveness'])
    # print(test_df['discourse_effectiveness'])

    ### KILL ARGUMENT TYPE, FOCUS ON EFFECTIVENESS:
    # for df in [train_df, validate_df, test_df]:
    #     df["discourse_type"] = 'O-Argument'
    ###

    ### KILL EFFECTIVENESS, FOCUS ON ARGUMENT TYPE
    # for df in [train_df, validate_df, test_df]:
    #     df['discourse_effectiveness'] = 'O-Effectivness'
    #     df['Ineffective'] = 0
    #     df['Effective'] = 0
    #     df['Adequate'] = 0
    ###

    # print(train_df['discourse_effectiveness'])
    # print(validate_df['discourse_effectiveness'])
    # print(test_df['discourse_effectiveness'])

    # print(train_df['Ineffective'])
    # print(train_df['Effective'])
    # print(train_df['Adequate'])
    # print(train_df.columns)
    # sys.exit(0)

    # Step 2: Preprocess Data and prepare output folders
    train = preprocess(args.data_dir, train_df)
    # train = train.head(3)
    # print(train)
    validate = preprocess(args.data_dir, validate_df)
    test = preprocess(args.data_dir, test_df)
    experiment_name = args.setting_name + '/multi_I_epoch_' + str(args.epochs) + '_maxlen_' + str(args.max_len)
    if os.path.isdir(experiment_name) is False:
        os.makedirs(experiment_name)

    # Step 3: Build Model and Tokenizer
    model, tokenizer = build_model_tokenizer_multi_task(args.model, 15, 7)

    train_data = FeedbackPrizeDataset_MultiTask(train, tokenizer, args.max_len)
    training_loader = load_data(train_data, args.batch_size)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    # Step 4: Train and Validation
    # evaluation_output = {}

    loss_overview = pd.DataFrame(columns=["epoch", "loss"])
    token_accuracy_overview = pd.DataFrame(columns=["epoch", "train_argument", "train_effectiveness", "val_argument", "val_effectiveness", "test_argument", "test_effectiveness"])
    span_argument_overview = pd.DataFrame()
    # span_argument_overview = pd.DataFrame(columns=["epoch", "TP_train", "TP_val", "TP_test", "FP_train", "FP_val", "FP_test", "FN_train", "FN_val", "FN_test", "precision_train", "precision_val", "precision_test", "recall_train", "recall_val", "recall_test", "F1_train", "F1_val", "F1_test"])
    span_effectiveness_overview = pd.DataFrame()
    # span_effectiveness_overview = pd.DataFrame(columns=["epoch", "TP_train", "TP_val", "TP_test", "FP_train", "FP_val", "FP_test", "FN_train", "FN_val", "FN_test", "precision_train", "precision_val", "precision_test", "recall_train", "recall_val", "recall_test", "F1_train", "F1_val", "F1_test"])

    token_accuracy_overview_noCondition = pd.DataFrame(columns=["epoch", "train_argument", "train_effectiveness", "val_argument", "val_effectiveness", "test_argument", "test_effectiveness"])
    span_argument_overview_noCondition = pd.DataFrame()
    span_effectiveness_overview_noCondition = pd.DataFrame()

    for epoch in range(args.epochs):

        if not os.path.exists(os.path.join(args.setting_name, str(epoch+1))):
            os.makedirs(os.path.join(args.setting_name, str(epoch+1)))

        print("Training epoch: "+str(epoch + 1))
        epoch_loss, train_token_acc_argument, train_token_acc_effectiveness = model_train_multi_task(training_loader, model, optimizer, device, args.max_norm)
        
        # Save epoch loss overview
        df_epoch_loss = pd.DataFrame(epoch_loss, columns=["loss"])
        df_epoch_loss["epoch"] = epoch+1
        loss_overview = pd.concat([loss_overview, df_epoch_loss])
        loss_overview.to_csv(os.path.join(args.setting_name, "loss_stats.csv"), index=None)

        #train_pred_argument, train_pred_effectiveness = model_predict_multi_task(device, model, args.max_len,tokenizer, train)
        #f1_argument_train, scores_argument_train = model_evaluate(train_pred_argument, train_df, "discourse_type")
        #f1_effectiveness_train, scores_effectiveness_train = model_evaluate(train_pred_effectiveness, train_df,"discourse_effectiveness")
        
        # Save model
        # torch.save(model, experiment_name + "/model_" + str(epoch + 1) + '.pt')

        token_acc_epoch, df_span_argument_epoch, df_span_effectiveness_epoch = do_evaluation(experiment_name=experiment_name, setting_name=args.setting_name, epoch=epoch ,device=device, model=model, max_len=args.max_len, tokenizer=tokenizer, train=train, validate=validate, test=test, train_df=train_df, validate_df=validate_df, test_df=test_df, use_greater_10_condition=True)
        # Save token accuracy statistics
        token_accuracy_overview = pd.concat([token_accuracy_overview, pd.DataFrame.from_dict(token_acc_epoch)])
        token_accuracy_overview.to_csv(os.path.join(args.setting_name, "token_accuracy_stats.csv"), index=None)
        # Save span classification statistics
        # # Argument
        span_argument_overview = pd.concat([span_argument_overview, df_span_argument_epoch])
        span_argument_overview = span_argument_overview.fillna(0)
        span_argument_overview.to_csv(os.path.join(args.setting_name, "span_argument_stats.csv"))
        # # Effectiveness
        span_effectiveness_overview = pd.concat([span_effectiveness_overview, df_span_effectiveness_epoch])
        span_effectiveness_overview = span_effectiveness_overview.fillna(0)
        span_effectiveness_overview.to_csv(os.path.join(args.setting_name, "span_effectiveness_stats.csv"))


        token_acc_epoch_noCondition, df_span_argument_epoch_noCondition, df_span_effectiveness_epoch_noCondition = do_evaluation(experiment_name=experiment_name, setting_name=args.setting_name, epoch=epoch ,device=device, model=model, max_len=args.max_len, tokenizer=tokenizer, train=train, validate=validate, test=test, train_df=train_df, validate_df=validate_df, test_df=test_df, use_greater_10_condition=False)
        # Save token accuracy statistics
        token_accuracy_overview_noCondition = pd.concat([token_accuracy_overview_noCondition, pd.DataFrame.from_dict(token_acc_epoch_noCondition)])
        token_accuracy_overview_noCondition.to_csv(os.path.join(args.setting_name, "token_accuracy_stats_noCondition.csv"), index=None)
        # Save span classification statistics
        # # Argument
        span_argument_overview_noCondition = pd.concat([span_argument_overview_noCondition, df_span_argument_epoch_noCondition])
        span_argument_overview_noCondition = span_argument_overview_noCondition.fillna(0)
        span_argument_overview_noCondition.to_csv(os.path.join(args.setting_name, "span_argument_stats_noCondition.csv"))
        # # Effectiveness
        span_effectiveness_overview_noCondition = pd.concat([span_effectiveness_overview_noCondition, df_span_effectiveness_epoch_noCondition])
        span_effectiveness_overview_noCondition = span_effectiveness_overview_noCondition.fillna(0)
        span_effectiveness_overview_noCondition.to_csv(os.path.join(args.setting_name, "span_effectiveness_stats_noCondition.csv"))


        #### ORIGINAL EVAL CODE BEFORE EXPORTING INTO METHOD

        # do_evaluation(use_greater_10_condition=True)

        # try:
        #     print(f"Evaluate on train: {epoch + 1}")
        #     train_pred_argument, train_pred_effectiveness, train_token_acc_argument, train_token_acc_effectiveness, cm_token_argument_train, cm_token_effectiveness_train = model_predict_multi_task(device, model, args.max_len, tokenizer, train, use_greater_10_condition=use_greater_10_condition)
            
        #     cm_token_argument_train.to_csv(os.path.join(args.setting_name, str(epoch+1), "cm_train_token_argument.csv"), index_label='Gold|Predicted')    
        #     cm_token_effectiveness_train.to_csv(os.path.join(args.setting_name, str(epoch+1), "cm_train_token_effectiveness.csv"), index_label='Gold|Predicted')
            
        #     print("Saving second matrix")
        #     # # Save confusion matrixes: Workaround because to_csv cuts off header
        #     # with open(os.path.join(args.setting_name, str(epoch+1), "cm_train_token_argument.csv"), 'w') as train_cm_argu:
        #     #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     #         train_cm_argu.write(str(cm_token_argument))
        #     # with open(os.path.join(args.setting_name, str(epoch+1), "cm_train_token_effectiveness.csv"), 'w') as train_cm_eff:
        #     #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     #         train_cm_eff.write(str(cm_token_effectiveness))

        #     f1_argument_train, scores_argument_train, cm_span_argument_train = model_evaluate(train_pred_argument, train_df, "discourse_type")
            
        #     print("First f1 done")
            
        #     f1_effectiveness_train, scores_effectiveness_train, cm_span_effectiveness_train = model_evaluate(train_pred_effectiveness, train_df,
        #                                                            "discourse_effectiveness")

        #     cm_span_argument_train.to_csv(os.path.join(args.setting_name, str(epoch+1), "cm_train_span_argument.csv"), index_label='Gold|Predicted')
        #     cm_span_effectiveness_train.to_csv(os.path.join(args.setting_name, str(epoch+1), "cm_train_span_effectiveness.csv"), index_label='Gold|Predicted')
        #     # with open(os.path.join(args.setting_name, str(epoch+1), "cm_train_span_argument.csv"), 'w') as train_cm_argu:
        #     #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     #         train_cm_argu.write(str(cm_span_argument))
        #     # with open(os.path.join(args.setting_name, str(epoch+1), "cm_train_span_effectiveness.csv"), 'w') as train_cm_eff:
        #     #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     #         train_cm_eff.write(str(cm_span_effectiveness))


        #     evaluation_output['train argument:' + str(epoch + 1)] = scores_argument_train
        #     evaluation_output['train effectiveness:' + str(epoch + 1)] = scores_effectiveness_train

        #     write_evaluation(evaluation_output, experiment_name + '/training_evaluation_epoch_'+str(epoch + 1)+'.csv')
        #     write_prediction(train_pred_argument, experiment_name + '/training_argument_prediction_epoch_'+str(epoch + 1)+'.csv')
        #     write_prediction(train_pred_effectiveness, experiment_name + '/training_effectiveness_prediction_epoch_' + str(epoch + 1) + '.csv')
        # except Exception as e:
        #     print(e)
        #     continue

        # try:
        #     print(f"Validating epoch: {epoch + 1}")
        #     validate_pred_argument, validate_pred_effectiveness, val_token_acc_argument, val_token_acc_effectiveness, cm_token_argument_val, cm_token_effectiveness_val = model_predict_multi_task(device, model, args.max_len, tokenizer, validate, use_greater_10_condition=use_greater_10_condition)
            
        #     cm_token_argument_val.to_csv(os.path.join(args.setting_name, str(epoch+1), "cm_val_token_argument.csv"), index_label='Gold|Predicted')
        #     cm_token_effectiveness_val.to_csv(os.path.join(args.setting_name, str(epoch+1), "cm_val_token_effectiveness.csv"), index_label='Gold|Predicted')
        #     # # Save confusion matrixes: Workaround because to_csv cuts off header
        #     # with open(os.path.join(args.setting_name, str(epoch+1), "cm_val_token_argument.csv"), 'w') as val_cm_argu:
        #     #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     #         val_cm_argu.write(str(cm_token_argument))
        #     # with open(os.path.join(args.setting_name, str(epoch+1), "cm_val_token_effectiveness.csv"), 'w') as val_cm_eff:
        #     #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     #         val_cm_eff.write(str(cm_token_effectiveness))

        #     f1_argument_val, scores_argument_val, cm_span_argument_val = model_evaluate(validate_pred_argument, validate_df, "discourse_type")
        #     f1_effectiveness_val, scores_effectiveness_val, cm_span_effectiveness_val = model_evaluate(validate_pred_effectiveness, validate_df,
        #                                                            "discourse_effectiveness")
            
        #     cm_span_argument_val.to_csv(os.path.join(args.setting_name, str(epoch+1), "cm_val_span_argument.csv"), index_label='Gold|Predicted')
        #     cm_span_effectiveness_val.to_csv(os.path.join(args.setting_name, str(epoch+1), "cm_val_span_effectiveness.csv"), index_label='Gold|Predicted')
        #     # with open(os.path.join(args.setting_name, str(epoch+1), "cm_val_span_argument.csv"), 'w') as test_cm_argu:
        #     #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     #         test_cm_argu.write(str(cm_span_argument))
        #     # with open(os.path.join(args.setting_name, str(epoch+1), "cm_val_span_effectiveness.csv"), 'w') as test_cm_eff:
        #     #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     #         test_cm_eff.write(str(cm_span_effectiveness))

        #     evaluation_output['validation argument:' + str(epoch + 1)] = scores_argument_val
        #     evaluation_output['validation effectiveness:' + str(epoch + 1)] = scores_effectiveness_val

        #     write_evaluation(evaluation_output, experiment_name + '/validation_evaluation_epoch_'+str(epoch + 1)+'.csv')
        #     write_prediction(validate_pred_argument, experiment_name + '/validation_argument_prediction_epoch_'+str(epoch + 1)+'.csv')
        #     write_prediction(validate_pred_effectiveness, experiment_name + '/validation_effectiveness_prediction_epoch_' + str(epoch + 1) + '.csv')
        # except Exception as e:
        #     print(e)
        #     continue

        # # write_evaluation(evaluation_output['validation effectiveness:' + str(epoch + 1)], experiment_name + '/validation_effectiveness_epoch_'+str(epoch + 1)+'.csv')



        # # STEP 5: Test
        # try:
        #     print("Test:")
        #     test_pred_argument, test_pred_effectiveness, test_token_acc_argument, test_token_acc_effectiveness, cm_token_argument_test, cm_token_effectiveness_test = model_predict_multi_task(device, model, args.max_len, tokenizer, test, use_greater_10_condition=use_greater_10_condition)

        #     # print("TEST ACCS", test_token_acc_argument, test_token_acc_effectiveness)

        #     cm_token_argument_test.to_csv(os.path.join(args.setting_name, str(epoch+1), "cm_test_token_argument.csv"), index_label='Gold|Predicted')
        #     cm_token_effectiveness_test.to_csv(os.path.join(args.setting_name, str(epoch+1), "cm_test_token_effectiveness.csv"), index_label='Gold|Predicted')
        #     # Save confusion matrixes: Workaround because to_csv cuts off header
        #     # with open(os.path.join(args.setting_name, str(epoch+1), "cm_test_token_argument.csv"), 'w') as test_cm_argu:
        #     #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     #         test_cm_argu.write(str(cm_token_argument))
        #     # with open(os.path.join(args.setting_name, str(epoch+1), "cm_test_token_effectiveness.csv"), 'w') as test_cm_eff:
        #     #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     #         test_cm_eff.write(str(cm_token_effectiveness))

        #     f1_argument, scores_argument_test, cm_span_argument_test = model_evaluate(test_pred_argument, test_df, "discourse_type")
        #     f1_effectiveness, scores_effectiveness_test, cm_span_effectiveness_test = model_evaluate(test_pred_effectiveness, test_df, "discourse_effectiveness")
            
        #     cm_span_argument_test.to_csv(os.path.join(args.setting_name, str(epoch+1), "cm_test_span_argument.csv"), index_label='Gold|Predicted')
        #     cm_span_effectiveness_test.to_csv(os.path.join(args.setting_name, str(epoch+1), "cm_test_span_effectiveness.csv"), index_label='Gold|Predicted')
        #     # with open(os.path.join(args.setting_name, str(epoch+1), "cm_test_span_argument.csv"), 'w') as test_cm_argu:
        #     #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     #         test_cm_argu.write(str(cm_span_argument))
        #     # with open(os.path.join(args.setting_name, str(epoch+1), "cm_test_span_effectiveness.csv"), 'w') as test_cm_eff:
        #     #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     #         test_cm_eff.write(str(cm_span_effectiveness))
            
        #     evaluation_output['test argument'] = scores_argument_test
        #     evaluation_output['test effectiveness'] = scores_effectiveness_test

        #     write_evaluation(evaluation_output, experiment_name + '/test_evaluation_argument.csv')
        #     write_prediction(test_pred_argument, experiment_name + '/test_argument_prediction.csv')
        #     write_prediction(test_pred_effectiveness, experiment_name + '/test_effectiveness_prediction.csv')
        # except Exception as e:
        #     print(e)
        # # write_evaluation(evaluation_output['test effectiveness'], experiment_name+ '/effectiveness_argument.csv')

        # # Save token accuracy statistics
        # token_acc_epoch = {"epoch": str(epoch+1), "train_argument": [train_token_acc_argument], "train_effectiveness": [train_token_acc_effectiveness], "val_argument": [val_token_acc_argument], "val_effectiveness": [val_token_acc_effectiveness], "test_argument": [test_token_acc_argument], "test_effectiveness": [test_token_acc_effectiveness]}
        # token_accuracy_overview = pd.concat([token_accuracy_overview, pd.DataFrame.from_dict(token_acc_epoch)])
        # token_accuracy_overview.to_csv(os.path.join(args.setting_name, "token_accuracy_stats.csv"), index=None)

        # # Save span classification statistics

        # # # Argument
        # # df_span_argument_epoch = reduce(lambda  left,right: pd.merge(left,right,left_index=True, right_index=True, suffixes=['_train', '_val'],
        #                                     # how='outer'), [pd.DataFrame.from_dict(scores_argument_train, orient='index'), pd.DataFrame.from_dict(scores_argument_val, orient='index'), pd.DataFrame.from_dict(scores_argument_test, orient='index')])
        # df_span_argument_epoch = pd.merge(left=pd.DataFrame.from_dict(scores_argument_train, orient='index'), right=pd.DataFrame.from_dict(scores_argument_val, orient='index'), left_index=True, right_index=True, suffixes=["_train", ""], how='outer')
        # df_span_argument_epoch = pd.merge(left=df_span_argument_epoch, right=pd.DataFrame.from_dict(scores_argument_test, orient='index'), left_index=True, right_index=True, suffixes=["_val", "_test"], how='outer')
        # df_span_argument_epoch["epoch"] = str(epoch+1)
        # span_argument_overview = pd.concat([span_argument_overview, df_span_argument_epoch])
        # span_argument_overview = span_argument_overview.fillna(0)
        # span_argument_overview.to_csv(os.path.join(args.setting_name, "span_argument_stats.csv"))

        # # # Effectiveness
        # df_span_effectiveness_epoch = pd.merge(left=pd.DataFrame.from_dict(scores_effectiveness_train, orient='index'), right=pd.DataFrame.from_dict(scores_effectiveness_val, orient='index'), left_index=True, right_index=True, suffixes=["_train", ""], how='outer')
        # df_span_effectiveness_epoch = pd.merge(left=df_span_effectiveness_epoch, right=pd.DataFrame.from_dict(scores_effectiveness_test, orient='index'), left_index=True, right_index=True, suffixes=["_val", "_test"], how='outer')
        # df_span_effectiveness_epoch["epoch"] = str(epoch+1)
        # span_effectiveness_overview = pd.concat([span_effectiveness_overview, df_span_effectiveness_epoch])
        # span_effectiveness_overview = span_effectiveness_overview.fillna(0)
        # span_effectiveness_overview.to_csv(os.path.join(args.setting_name, "span_effectiveness_stats.csv"))

        # print("EPOCH", epoch+1)
        # print("SCORES", epoch+1, scores_argument_train)
        # print("EPOCH", epoch+1)