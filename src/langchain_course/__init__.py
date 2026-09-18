from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama

load_dotenv()


def main() -> None:
    print("Hello from langchain-course!")
    information = """Abū Mūsā Jābir ibn Ḥayyān (Arabic: أَبو موسى جابِر بِن حَيّان, variously called al-Ṣūfī, al-Azdī, al-Kūfī, or al-Ṭūsī), died c. 806−816, is the purported author of a large number of works in Arabic, often called the Jabirian corpus. The c. 215 treatises that survive today mainly deal with alchemy and chemistry, magic, and Shi'ite religious philosophy. However, the original scope of the corpus was vast, covering a wide range of topics ranging from cosmology, astronomy and astrology, over medicine, pharmacology, zoology and botany, to metaphysics, logic, and grammar.

The works attributed to Jabir, which are tentatively dated to c. 850 – c. 950,[1] contain the oldest known systematic classification of chemical substances, and the oldest known instructions for deriving an inorganic compound (sal ammoniac or ammonium chloride) from organic substances (such as plants, blood, and hair) by chemical means.[2] His works also contain one of the earliest known versions of the sulfur-mercury theory of metals, a mineralogical theory that would remain dominant until the 18th century.[3]

A significant part of Jabir's writings deal with a philosophical theory known as "the science of the balance" (Arabic: ʿilm al-mīzān), which was aimed at reducing all phenomena (including material substances and their elements) to a system of measures and quantitative proportions. The Jabirian works also contain some of the earliest preserved Shi'ite imamological doctrines, which Jabir presented as deriving from his purported master, the Shi'ite Imam Jaʿfar al-Ṣādiq (died 765).

As early as the 10th century, the identity and exact corpus of works of Jabir was in dispute in Islamic scholarly circles. The authorship of all these works by a single figure, and even the existence of a historical Jabir, are also doubted by modern scholars. Instead, Jabir ibn Hayyan is generally thought to have been a pseudonym used by an anonymous school of Shi'ite alchemists writing in the late 9th and early 10th centuries.

Some Arabic Jabirian works (e.g., The Great Book of Mercy, and The Book of Seventy) were translated into Latin under the Latinized name Geber, and in 13th-century Europe an anonymous writer, usually referred to as pseudo-Geber, started to produce alchemical and metallurgical writings under this name.[4]

"""

    summary_template = """given the information {information} about a person. I want you to create:
1. A short summary
2. two interesting facts about them
"""
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    #llm = ChatDeepSeek(temperature=0, model="deepseek-v4-flash")
    llm = ChatOllama(temperature=0, model="gpt-oss:20b")
    chain = (
        summary_prompt_template | llm
    )  # runnable object, langChain expression language
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
