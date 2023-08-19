from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
import transformers
import evaluate
from transformers import TrainingArguments, Trainer
import math
import argparse
from distutils.util import strtobool


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BERTweet Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_checkpoint", type=str, help="Model name", default='neuralmind/bert-base-portuguese-cased'
    )
    parser.add_argument(
        "--tokenizer_checkpoint", type=str, help="", default='neuralmind/bert-base-portuguese-cased'
    )
    parser.add_argument(
        "--chunk_size", type=int, help="", default=128
    )
    parser.add_argument(
        "--batch_size", type=int, help="Evaluation Batch Size", default=32
    )
    parser.add_argument(
        "--learning_rate", type=float, help="", default=2e-5
    )
    parser.add_argument(
        "--weight_decay", type=float, help="", default=0.01
    )
    parser.add_argument(
        "--output_dir", type=str, help="Where to save results", default="./BERTweetBR"
    )
    parser.add_argument(
        "--logging_dir", type=str, help="Where to save results", default="./BERTweetBR_logs"
    )
    parser.add_argument(
        "--evaluation_strategy", type=str, help="Where to save results", default="steps"
    )
    parser.add_argument(
        "--fp16", type=lambda x: bool(strtobool(x)), help="Language code", default=False
    )
    parser.add_argument(
        "--overwrite_output_dir", type=lambda x: bool(strtobool(x)), help="Language code", default=True
    )
    parser.add_argument(
        "--is_code_test", type=lambda x: bool(strtobool(x)), help="Language code", default=True
    )
    # TO DO - Help comments in the arguments
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    # Pega o model
    model = AutoModelForMaskedLM.from_pretrained(args.model_checkpoint)

    # Pega o tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_checkpoint)

    # Pega o Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    # Prepara datasets
    # TO DO - ter uma pasta de dados de treino e teste e passar a lista dos arquivos como parametro - os lib

    raw_dataset = load_dataset('text', data_files={
        'train': ['./tweets/text/text_1.txt', './tweets/text/text_2.txt', './tweets/text/text_3.txt',
                  './tweets/text/text_4.txt', './tweets/text/text_5.txt', './tweets/text/text_6.txt',
                  './tweets/text/text_7.txt']})

    # Split do dataset - TO DO - remover isso
    if args.is_code_test:
        train_size = 1000
        test_size = int(0.1 * train_size)
        raw_dataset = raw_dataset["train"].train_test_split(train_size=train_size, test_size=test_size, seed=42)

    # Funcao para tokenizacao
    def tokenize_function(examples):
        result = tokenizer(examples["text"])
        if tokenizer.is_fast:  # TO DO - verificar se isso é necessário
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result

    # Tokenizando datasets
    final_dataset = raw_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    # Carrega metrica de perplexidade
    metric = evaluate.load("perplexity")

    # Muda verbosidade do transformers
    transformers.logging.set_verbosity_info()

    # Mostra log a cada step definido abaixo
    logging_steps = len(final_dataset["train"]) // args.batch_size

    # Prepara os TrainingArguments

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        evaluation_strategy=args.evaluation_strategy,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        fp16=args.fp16,
        logging_steps=logging_steps,
    )

    # Prepara o Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=final_dataset["train"],
        eval_dataset=final_dataset["test"],
        data_collator=data_collator,
    )

    # Coleta perplexidade antes de treinar, somente avaliando
    eval_results = trainer.evaluate()
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    initial_perplexity = math.exp(eval_results['eval_loss'])

    # Treina
    train_result = trainer.train()

    # Coleta perplexidade apos treinar, avaliando
    eval_results = trainer.evaluate()
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    # Coletando metricas do resultado de train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(final_dataset["train"])

    # Save train results
    trainer.log_metrics("all", metrics)
    trainer.save_metrics("all", metrics)

    # Cria log do historico do obj do Trainer
    with open(str(args.logging_dir) + '/trainer_logs.txt', 'w') as f:
        for obj in trainer.state.log_history:
            f.write(str(obj))
            f.write('\n')
        f.write('\n\n\n')
        f.write(str(metrics))
        f.write('\n\n\n')
        f.write('Initial Perplexity = ' + str(initial_perplexity))
        f.write('\n')
        f.write('Final Perplexity = ' + str(math.exp(eval_results['eval_loss'])))

    # Salva modelo treinado
    trainer.save_model(args.output_dir)

if __name__ == '__main__':
    main()