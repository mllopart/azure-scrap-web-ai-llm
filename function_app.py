import logging
import os

import azure.functions as func
import tiktoken
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


@app.route(route="summarize", methods=["GET"], auth_level=func.AuthLevel.FUNCTION)
def summarize(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger summarize function processed a request.")

    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    model_max_tokens = 4097
    verbose = False
    model_name = "gpt-3.5-turbo"

    url = req.params.get("url")
    if not url:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            url = req_body.get("url")

    if url:
        try:
            loader = AsyncChromiumLoader([url])
            html = loader.load()

            # Transform
            bs_transformer = BeautifulSoupTransformer()
            docs_transformed = bs_transformer.transform_documents(
                html, tags_to_extract=["span"]
            )

            text = docs_transformed[0].page_content

            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                model_name=model_name
            )
            texts = text_splitter.split_text(text)
            docs = [Document(page_content=t) for t in texts]
            logging.info(f"docs: {len(docs)}")

            llm = ChatOpenAI(
                temperature=0, openai_api_key=OPENAI_API_KEY, model_name=model_name
            )

            prompt_template = """
            you are a knowledge expert that your job is to summarize clearly.
            Write a concise and easy to understand in a clear way that is easy to understandsummary of the following:
           
            {text}
            
            CONSCISE SUMMARY:
            """

            prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
            num_tokens = num_tokens_from_string(text, model_name)
            logging.info(f"num_tokens: {num_tokens}")

            if num_tokens < model_max_tokens:
                chain = load_summarize_chain(
                    llm, chain_type="stuff", prompt=prompt, verbose=verbose
                )
            else:
                chain = load_summarize_chain(
                    llm,
                    chain_type="map_reduce",
                    map_prompt=prompt,
                    combine_prompt=prompt,
                    verbose=verbose,
                )

            summary = chain.run(docs)

            return func.HttpResponse(
                summary,
                status_code=200,
            )
        except Exception as e:
            return func.HttpResponse(
                f"Error: {e}",
                status_code=400,
            )

    else:
        return func.HttpResponse(
            "Please pass a url on the query string or in the request body",
            status_code=400,
        )


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
