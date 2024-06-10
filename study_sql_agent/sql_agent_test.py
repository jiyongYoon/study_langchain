from dotenv import load_dotenv
import os
import openai


load_dotenv()
print(os.environ['PROJECT_ROOT'])

### llm을 활용하여 sql문 생성 및 쿼리 동작

from langchain_community.llms import OpenAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain


db = SQLDatabase.from_uri("sqlite:///chinook.db")
llm = OpenAI(
    temperature=0,
    verbose=True,
    api_key=os.environ['OPEN_API_KEY'],
)
db_chain = SQLDatabaseChain.from_llm(
    llm,
    db,
    verbose=True # 어떤식으로 동작이 되는지 확인
)


# print(db_chain.run("몇명의 직원이 있어?"))
"""
> Entering new SQLDatabaseChain chain...
몇명의 직원이 있어?
SQLQuery:SELECT COUNT(*) FROM employees
SQLResult: [(8,)]
Answer:8
> Finished chain.
8
"""

### 단계 구분 -> sql 쿼리문만 생성

from langchain.chains import create_sql_query_chain
from langchain_community.chat_models import ChatOpenAI


chain = create_sql_query_chain( # 쿼리 작성 체인 생성
    ChatOpenAI(
        temperature=0,
        api_key=os.environ['OPEN_API_KEY'],
    ),
    db
)

# generated_sql = chain.invoke({"question": "How many employees are there"})
# print(generated_sql)
# # SELECT COUNT("EmployeeId") AS "TotalEmployees" FROM employees
# print(db.run(generated_sql))


### SQL Agent 실습

from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType

### agnet에게 sql_database_toolkit을 넣어주어 동작시키기

agent_executor = create_sql_agent(
    llm=OpenAI(
        temperature=0,
        api_key=os.environ['OPEN_API_KEY'],
    ),
    toolkit=SQLDatabaseToolkit(
        db=db,
        llm=OpenAI(
            temperature=0,
            api_key=os.environ['OPEN_API_KEY'],
        )),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, # 예제 없이 수행
)


# agent_executor.run(
#     "국가별 총 매출을 나열합니다. 어느 국가의 고객이 가장 많이 지출했나요?"
# )
"""
> Entering new SQL Agent Executor chain...
 We need to find the total sales for each country and then determine which country has the highest total sales.
Action: sql_db_query
Action Input: SELECT country, SUM(sales) AS total_sales FROM customers GROUP BY country ORDER BY total_sales DESC LIMIT 1Error: (sqlite3.OperationalError) no such column: sales
[SQL: SELECT country, SUM(sales) AS total_sales FROM customers GROUP BY country ORDER BY total_sales DESC LIMIT 1]
(Background on this error at: https://sqlalche.me/e/20/e3q8)We need to check the table schema to see if there is a column named 'sales'.
Action: sql_db_schema
Action Input: customers
CREATE TABLE customers (
	"CustomerId" INTEGER NOT NULL, 
	"FirstName" NVARCHAR(40) NOT NULL, 
	"LastName" NVARCHAR(20) NOT NULL, 
	"Company" NVARCHAR(80), 
	"Address" NVARCHAR(70), 
	"City" NVARCHAR(40), 
	"State" NVARCHAR(40), 
	"Country" NVARCHAR(40), 
	"PostalCode" NVARCHAR(10), 
	"Phone" NVARCHAR(24), 
	"Fax" NVARCHAR(24), 
	"Email" NVARCHAR(60) NOT NULL, 
	"SupportRepId" INTEGER, 
	PRIMARY KEY ("CustomerId"), 
	FOREIGN KEY("SupportRepId") REFERENCES employees ("EmployeeId")
)

/*
3 rows from customers table:
CustomerId	FirstName	LastName	Company	Address	City	State	Country	PostalCode	Phone	Fax	Email	SupportRepId
1	Luís	Gonçalves	Embraer - Empresa Brasileira de Aeronáutica S.A.	Av. Brigadeiro Faria Lima, 2170	São José dos Campos	SP	Brazil	12227-000	+55 (12) 3923-5555	+55 (12) 3923-5566	luisg@embraer.com.br	3
2	Leonie	Köhler	None	Theodor-Heuss-Straße 34	Stuttgart	None	Germany	70174	+49 0711 2842222	None	leonekohler@surfeu.de	5
3	François	Tremblay	None	1498 rue Bélanger	Montréal	QC	Canada	H2G 1A7	+1 (514) 721-4711	None	ftremblay@gmail.com	3
*/ We can see that there is no 'sales' column in the customers table. We need to check the table schema for the correct column name.
Action: sql_db_schema
Action Input: invoices
CREATE TABLE invoices (
	"InvoiceId" INTEGER NOT NULL, 
	"CustomerId" INTEGER NOT NULL, 
	"InvoiceDate" DATETIME NOT NULL, 
	"BillingAddress" NVARCHAR(70), 
	"BillingCity" NVARCHAR(40), 
	"BillingState" NVARCHAR(40), 
	"BillingCountry" NVARCHAR(40), 
	"BillingPostalCode" NVARCHAR(10), 
	"Total" NUMERIC(10, 2) NOT NULL, 
	PRIMARY KEY ("InvoiceId"), 
	FOREIGN KEY("CustomerId") REFERENCES customers ("CustomerId")
)

/*
3 rows from invoices table:
InvoiceId	CustomerId	InvoiceDate	BillingAddress	BillingCity	BillingState	BillingCountry	BillingPostalCode	Total
1	2	2009-01-01 00:00:00	Theodor-Heuss-Straße 34	Stuttgart	None	Germany	70174	1.98
2	4	2009-01-02 00:00:00	Ullevålsveien 14	Oslo	None	Norway	0171	3.96
3	8	2009-01-03 00:00:00	Grétrystraat 63	Brussels	None	Belgium	1000	5.94
*/ We can see that there is a 'Total' column in the invoices table. We can use this column to find the total sales for each country.
Action: sql_db_query
Action Input: SELECT BillingCountry AS country, SUM(Total) AS total_sales FROM invoices GROUP BY country ORDER BY total_sales DESC LIMIT 1[('USA', 523.0600000000003)] We can see that the USA has the highest total sales.
Final Answer: The USA has the highest total sales.

> Finished chain.
"""

# agent_executor.run(
#     "playlisttrack 테이블에 대해서 설명해줄래?"
# )
"""
> Entering new SQL Agent Executor chain...
 I should use sql_db_schema to get the schema and sample rows for the playlisttrack table.
Action: sql_db_schema
Action Input: playlisttrackError: table_names {'playlisttrack'} not found in databaseI should use sql_db_list_tables to get a list of tables in the database and check if playlisttrack is included.
Action: sql_db_list_tables
Action Input: albums, artists, customers, employees, genres, invoice_items, invoices, media_types, playlist_track, playlists, tracksI should use sql_db_schema again with the correct spelling of playlist_track.
Action: sql_db_schema
Action Input: playlist_track
CREATE TABLE playlist_track (
	"PlaylistId" INTEGER NOT NULL, 
	"TrackId" INTEGER NOT NULL, 
	PRIMARY KEY ("PlaylistId", "TrackId"), 
	FOREIGN KEY("TrackId") REFERENCES tracks ("TrackId"), 
	FOREIGN KEY("PlaylistId") REFERENCES playlists ("PlaylistId")
)

/*
3 rows from playlist_track table:
PlaylistId	TrackId
1	3402
1	3389
1	3390
*/I now know the final answer.
Final Answer: 
The playlisttrack table has two columns, PlaylistId and TrackId, and is used to store the relationship between playlists and tracks. 
It has a composite primary key consisting of both columns and foreign key constraints referencing the playlists and tracks tables. 
Sample rows from the table show the relationship between a playlist with id 1 and three tracks with ids 3402, 3389, and 3390.

> Finished chain.
"""

## SQL 툴킷 확장

### Dynamic Few shot 예제

# 도메인에 해당하는 지식을 활용하여 llm이 미리 참고할 자료들을 추가함
few_shots = {
    "List all artists.": "SELECT * FROM artists;",
    "Find all albums for the artist 'AC/DC'.": "SELECT * FROM albums WHERE ArtistId = (SELECT ArtistId FROM artists WHERE Name = 'AC/DC');",
    "List all tracks in the 'Rock' genre.": "SELECT * FROM tracks WHERE GenreId = (SELECT GenreId FROM genres WHERE Name = 'Rock');",
    "Find the total duration of all tracks.": "SELECT SUM(Milliseconds) FROM tracks;",
    "List all customers from Canada.": "SELECT * FROM customers WHERE Country = 'Canada';",
    "How many tracks are there in the album with ID 5?": "SELECT COUNT(*) FROM tracks WHERE AlbumId = 5;",
    "Find the total number of invoices.": "SELECT COUNT(*) FROM invoices;",
    "List all tracks that are longer than 5 minutes.": "SELECT * FROM tracks WHERE Milliseconds > 300000;",
    "Who are the top 5 customers by total purchase?": "SELECT CustomerId, SUM(Total) AS TotalPurchase FROM invoices GROUP BY CustomerId ORDER BY TotalPurchase DESC LIMIT 5;",
    "Which albums are from the year 2000?": "SELECT * FROM albums WHERE strftime('%Y', ReleaseDate) = '2000';",
    "How many employees are there": 'SELECT COUNT(*) FROM "employee"',
}

# 벡터스토어에 저장한 후 retriever를 통해 검색기를 생성

from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings(
    api_key=os.environ['OPEN_API_KEY'],
)

few_shot_docs = [
    # Document 객체에 question이 내용으로, sql_query가 metadata로 들어가게 됨
    Document(
        page_content=question,
        metadata={"sql_query": few_shots[question]} # few_shots의 question에 해당하는 값을 sql_query로 저장
    )
    for question in few_shots.keys()
]
vector_db = FAISS.from_documents(few_shot_docs, embeddings)
retriever = vector_db.as_retriever()

#

from langchain.agents.agent_toolkits import create_retriever_tool

tool_description = """
이 도구는 유사한 예시를 이해하여 사용자 질문에 적용하는 데 도움이 됩니다.
이 도구에 입력하는 내용은 사용자 질문이어야 합니다.
"""

retriever_tool = create_retriever_tool(
    retriever,
    name="sql_get_similar_examples",
    description=tool_description,
)

custom_tool_list = [retriever_tool]


llm = ChatOpenAI(
    model_name="gpt-4-1106-preview",
    temperature=0,
    api_key=os.environ['OPEN_API_KEY'],
)

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=llm
)

custom_suffix = """
먼저 제가 알고 있는 비슷한 예제를 가져와야 합니다.
예제가 쿼리를 구성하기에 충분하다면 쿼리를 작성할 수 있습니다.
그렇지 않으면 데이터베이스의 테이블을 살펴보고 쿼리할 수 있는 항목을 확인할 수 있습니다.
그런 다음 가장 관련성이 높은 테이블의 스키마를 쿼리해야 합니다.
"""

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    extra_tools=custom_tool_list,
    suffix=custom_suffix,
)

# agent.run("How many employees do we have?")
"""
> Entering new SQL Agent Executor chain...

Invoking: `sql_get_similar_examples` with `{'query': 'How many employees do we have?'}`


How many employees are there <-------- few_shot으로 제공한 비슷한 내용을 찾음

Find the total number of invoices.

Who are the top 5 customers by total purchase?

List all customers from Canada.
Invoking: `sql_db_list_tables` with `{'tool_input': ''}`


albums, artists, customers, employees, genres, invoice_items, invoices, media_types, playlist_track, playlists, tracks
Invoking: `sql_db_schema` with `{'table_names': 'employees'}`



CREATE TABLE employees (
	"EmployeeId" INTEGER NOT NULL, 
	"LastName" NVARCHAR(20) NOT NULL, 
	"FirstName" NVARCHAR(20) NOT NULL, 
	"Title" NVARCHAR(30), 
	"ReportsTo" INTEGER, 
	"BirthDate" DATETIME, 
	"HireDate" DATETIME, 
	"Address" NVARCHAR(70), 
	"City" NVARCHAR(40), 
	"State" NVARCHAR(40), 
	"Country" NVARCHAR(40), 
	"PostalCode" NVARCHAR(10), 
	"Phone" NVARCHAR(24), 
	"Fax" NVARCHAR(24), 
	"Email" NVARCHAR(60), 
	PRIMARY KEY ("EmployeeId"), 
	FOREIGN KEY("ReportsTo") REFERENCES employees ("EmployeeId")
)

/*
3 rows from employees table:
EmployeeId	LastName	FirstName	Title	ReportsTo	BirthDate	HireDate	Address	City	State	Country	PostalCode	Phone	Fax	Email
1	Adams	Andrew	General Manager	None	1962-02-18 00:00:00	2002-08-14 00:00:00	11120 Jasper Ave NW	Edmonton	AB	Canada	T5K 2N1	+1 (780) 428-9482	+1 (780) 428-3457	andrew@chinookcorp.com
2	Edwards	Nancy	Sales Manager	1	1958-12-08 00:00:00	2002-05-01 00:00:00	825 8 Ave SW	Calgary	AB	Canada	T2P 2T3	+1 (403) 262-3443	+1 (403) 262-3322	nancy@chinookcorp.com
3	Peacock	Jane	Sales Support Agent	2	1973-08-29 00:00:00	2002-04-01 00:00:00	1111 6 Ave SW	Calgary	AB	Canada	T2P 5M5	+1 (403) 262-3443	+1 (403) 262-6712	jane@chinookcorp.com
*/
Invoking: `sql_db_query_checker` with `{'query': 'SELECT COUNT(*) AS NumberOfEmployees FROM employees'}`


The provided SQLite query does not exhibit any of the common mistakes listed. It is a simple query that counts the number of rows in the 'employees' table. Therefore, the original query is correct as it stands:

```sql
SELECT COUNT(*) AS NumberOfEmployees FROM employees
```
Invoking: `sql_db_query` with `{'query': 'SELECT COUNT(*) AS NumberOfEmployees FROM employees'}`


[(8,)]We have a total of 8 employees.

> Finished chain.
"""

# 고유명사의 맞춤법 오류 찾기 및 수정하기

# print(db.table_info)

import ast
import re


def run_query_save_results(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el] # 결과를 리스트에 담기
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res] # 공백을 제거하기
    return res


artists = run_query_save_results(db, "SELECT name FROM artists")
albums = run_query_save_results(db, "SELECT title FROM albums")

print(artists)
print(albums)
"""
['AC/DC', 'Accept', 'Aerosmith', 'Alanis Morissette', 'Alice In Chains', 'Antônio Carlos Jobim', 'Apocalyptica', 'Audioslave', 'BackBeat', 'Billy Cobham', 'Black Label Society', 'Black Sabbath', 'Body Count', 'Bruce Dickinson', 'Buddy Guy', 'Caetano Veloso', 'Chico Buarque', 'Chico Science & Nação Zumbi', 'Cidade Negra', 'Cláudio Zoli', 'Various Artists', 'Led Zeppelin', 'Frank Zappa & Captain Beefheart', 'Marcos Valle', 'Milton Nascimento & Bebeto', 'Azymuth', 'Gilberto Gil', 'João Gilberto', 'Bebel Gilberto', 'Jorge Vercilo', 'Baby Consuelo', 'Ney Matogrosso', 'Luiz Melodia', 'Nando Reis', 'Pedro Luís & A Parede', 'O Rappa', 'Ed Motta', 'Banda Black Rio', 'Fernanda Porto', 'Os Cariocas', 'Elis Regina', 'Milton Nascimento', 'A Cor Do Som', 'Kid Abelha', 'Sandra De Sá', 'Jorge Ben', 'Hermeto Pascoal', 'Barão Vermelho', 'Edson, DJ Marky & DJ Patife Featuring Fernanda Porto', 'Metallica', 'Queen', 'Kiss', 'Spyro Gyra', 'Green Day', 'David Coverdale', 'Gonzaguinha', 'Os Mutantes', 'Deep Purple', 'Santana', 'Santana Feat. Dave Matthews', 'Santana Feat. Everlast', 'Santana Feat. Rob Thomas', 'Santana Feat. Lauryn Hill & Cee-Lo', 'Santana Feat. The Project G&B', 'Santana Feat. Maná', 'Santana Feat. Eagle-Eye Cherry', 'Santana Feat. Eric Clapton', 'Miles Davis', 'Gene Krupa', 'Toquinho & Vinícius', 'Vinícius De Moraes & Baden Powell', 'Vinícius De Moraes', 'Vinícius E Qurteto Em Cy', 'Vinícius E Odette Lara', 'Vinicius, Toquinho & Quarteto Em Cy', 'Creedence Clearwater Revival', 'Cássia Eller', 'Def Leppard', 'Dennis Chambers', 'Djavan', 'Eric Clapton', 'Faith No More', 'Falamansa', 'Foo Fighters', 'Frank Sinatra', 'Funk Como Le Gusta', 'Godsmack', "Guns N' Roses", 'Incognito', 'Iron Maiden', 'James Brown', 'Jamiroquai', 'JET', 'Jimi Hendrix', 'Joe Satriani', 'Jota Quest', 'João Suplicy', 'Judas Priest', 'Legião Urbana', 'Lenny Kravitz', 'Lulu Santos', 'Marillion', 'Marisa Monte', 'Marvin Gaye', 'Men At Work', 'Motörhead', 'Motörhead & Girlschool', 'Mônica Marianno', 'Mötley Crüe', 'Nirvana', 'O Terço', 'Olodum', 'Os Paralamas Do Sucesso', 'Ozzy Osbourne', 'Page & Plant', 'Passengers', "Paul D'Ianno", 'Pearl Jam', 'Peter Tosh', 'Pink Floyd', 'Planet Hemp', 'R.E.M. Feat. Kate Pearson', 'R.E.M. Feat. KRS-One', 'R.E.M.', 'Raimundos', 'Raul Seixas', 'Red Hot Chili Peppers', 'Rush', 'Simply Red', 'Skank', 'Smashing Pumpkins', 'Soundgarden', 'Stevie Ray Vaughan & Double Trouble', 'Stone Temple Pilots', 'System Of A Down', 'Terry Bozzio, Tony Levin & Steve Stevens', 'The Black Crowes', 'The Clash', 'The Cult', 'The Doors', 'The Police', 'The Rolling Stones', 'The Tea Party', 'The Who', 'Tim Maia', 'Titãs', 'Battlestar Galactica', 'Heroes', 'Lost', 'U2', 'UB40', 'Van Halen', 'Velvet Revolver', 'Whitesnake', 'Zeca Pagodinho', 'The Office', 'Dread Zeppelin', 'Battlestar Galactica (Classic)', 'Aquaman', 'Christina Aguilera featuring BigElf', "Aerosmith & Sierra Leone's Refugee Allstars", 'Los Lonely Boys', 'Corinne Bailey Rae', 'Dhani Harrison & Jakob Dylan', 'Jackson Browne', 'Avril Lavigne', 'Big & Rich', "Youssou N'Dour", 'Black Eyed Peas', 'Jack Johnson', 'Ben Harper', 'Snow Patrol', 'Matisyahu', 'The Postal Service', 'Jaguares', 'The Flaming Lips', "Jack's Mannequin & Mick Fleetwood", 'Regina Spektor', 'Scorpions', 'House Of Pain', 'Xis', 'Nega Gizza', 'Gustavo & Andres Veiga & Salazar', 'Rodox', 'Charlie Brown Jr.', 'Pedro Luís E A Parede', 'Los Hermanos', 'Mundo Livre S/A', 'Otto', 'Instituto', 'Nação Zumbi', 'DJ Dolores & Orchestra Santa Massa', 'Seu Jorge', 'Sabotage E Instituto', 'Stereo Maracana', 'Cake', 'Aisha Duo', 'Habib Koité and Bamada', 'Karsh Kale', 'The Posies', 'Luciana Souza/Romero Lubambo', 'Aaron Goldberg', 'Nicolaus Esterhazy Sinfonia', 'Temple of the Dog', 'Chris Cornell', 'Alberto Turco & Nova Schola Gregoriana', 'Richard Marlow & The Choir of Trinity College, Cambridge', 'English Concert & Trevor Pinnock', 'Anne-Sophie Mutter, Herbert Von Karajan & Wiener Philharmoniker', 'Hilary Hahn, Jeffrey Kahane, Los Angeles Chamber Orchestra & Margaret Batjer', 'Wilhelm Kempff', 'Yo-Yo Ma', 'Scholars Baroque Ensemble', 'Academy of St. Martin in the Fields & Sir Neville Marriner', 'Academy of St. Martin in the Fields Chamber Ensemble & Sir Neville Marriner', 'Berliner Philharmoniker, Claudio Abbado & Sabine Meyer', 'Royal Philharmonic Orchestra & Sir Thomas Beecham', 'Orchestre Révolutionnaire et Romantique & John Eliot Gardiner', 'Britten Sinfonia, Ivor Bolton & Lesley Garrett', 'Chicago Symphony Chorus, Chicago Symphony Orchestra & Sir Georg Solti', 'Sir Georg Solti & Wiener Philharmoniker', 'Academy of St. Martin in the Fields, John Birch, Sir Neville Marriner & Sylvia McNair', 'London Symphony Orchestra & Sir Charles Mackerras', 'Barry Wordsworth & BBC Concert Orchestra', 'Herbert Von Karajan, Mirella Freni & Wiener Philharmoniker', 'Eugene Ormandy', 'Luciano Pavarotti', 'Leonard Bernstein & New York Philharmonic', 'Boston Symphony Orchestra & Seiji Ozawa', 'Aaron Copland & London Symphony Orchestra', 'Ton Koopman', 'Sergei Prokofiev & Yuri Temirkanov', 'Chicago Symphony Orchestra & Fritz Reiner', 'Orchestra of The Age of Enlightenment', 'Emanuel Ax, Eugene Ormandy & Philadelphia Orchestra', 'James Levine', 'Berliner Philharmoniker & Hans Rosbaud', 'Maurizio Pollini', 'Academy of St. Martin in the Fields, Sir Neville Marriner & William Bennett', 'Gustav Mahler', 'Felix Schmidt, London Symphony Orchestra & Rafael Frühbeck de Burgos', 'Edo de Waart & San Francisco Symphony', 'Antal Doráti & London Symphony Orchestra', 'Choir Of Westminster Abbey & Simon Preston', 'Michael Tilson Thomas & San Francisco Symphony', 'Chor der Wiener Staatsoper, Herbert Von Karajan & Wiener Philharmoniker', "The King's Singers", 'Berliner Philharmoniker & Herbert Von Karajan', 'Sir Georg Solti, Sumi Jo & Wiener Philharmoniker', "Christopher O'Riley", 'Fretwork', 'Amy Winehouse', 'Calexico', 'Otto Klemperer & Philharmonia Orchestra', 'Yehudi Menuhin', 'Philharmonia Orchestra & Sir Neville Marriner', 'Academy of St. Martin in the Fields, Sir Neville Marriner & Thurston Dart', 'Les Arts Florissants & William Christie', 'The  Cellists of The Berlin Philharmonic', 'Adrian Leaper & Doreen de Feis', 'Roger Norrington, London Classical Players', "Charles Dutoit & L'Orchestre Symphonique de Montréal", 'Equale Brass Ensemble, John Eliot Gardiner & Munich Monteverdi Orchestra and Choir', "Kent Nagano and Orchestre de l'Opéra de Lyon", 'Julian Bream', 'Martin Roscoe', 'Göteborgs Symfoniker & Neeme Järvi', 'Itzhak Perlman', 'Michele Campanella', 'Gerald Moore', 'Mela Tenenbaum, Pro Musica Prague & Richard Kapp', 'Emerson String Quartet', 'C. Monteverdi, Nigel Rogers - Chiaroscuro; London Baroque; London Cornett & Sackbu', 'Nash Ensemble', 'Philip Glass Ensemble']
['For Those About To Rock We Salute You', 'Balls to the Wall', 'Restless and Wild', 'Let There Be Rock', 'Big Ones', 'Jagged Little Pill', 'Facelift', 'Warner  Anos', 'Plays Metallica By Four Cellos', 'Audioslave', 'Out Of Exile', 'BackBeat Soundtrack', 'The Best Of Billy Cobham', 'Alcohol Fueled Brewtality Live! [Disc ]', 'Alcohol Fueled Brewtality Live! [Disc ]', 'Black Sabbath', 'Black Sabbath Vol.  (Remaster)', 'Body Count', 'Chemical Wedding', 'The Best Of Buddy Guy - The Millenium Collection', 'Prenda Minha', 'Sozinho Remix Ao Vivo', 'Minha Historia', 'Afrociberdelia', 'Da Lama Ao Caos', 'Acústico MTV [Live]', 'Cidade Negra - Hits', 'Na Pista', 'Axé Bahia', 'BBC Sessions [Disc ] [Live]', 'Bongo Fury', 'Carnaval', 'Chill: Brazil (Disc )', 'Chill: Brazil (Disc )', 'Garage Inc. (Disc )', 'Greatest Hits II', 'Greatest Kiss', 'Heart of the Night', 'International Superhits', 'Into The Light', 'Meus Momentos', 'Minha História', 'MK III The Final Concerts [Disc ]', 'Physical Graffiti [Disc ]', 'Sambas De Enredo', 'Supernatural', 'The Best of Ed Motta', 'The Essential Miles Davis [Disc ]', 'The Essential Miles Davis [Disc ]', 'The Final Concerts (Disc )', "Up An' Atom", 'Vinícius De Moraes - Sem Limite', 'Vozes do MPB', 'Chronicle, Vol.', 'Chronicle, Vol.', 'Cássia Eller - Coleção Sem Limite [Disc ]', 'Cássia Eller - Sem Limite [Disc ]', 'Come Taste The Band', 'Deep Purple In Rock', 'Fireball', "Knocking at Your Back Door: The Best Of Deep Purple in the 's", 'Machine Head', 'Purpendicular', 'Slaves And Masters', 'Stormbringer', 'The Battle Rages On', "Vault: Def Leppard's Greatest Hits", 'Outbreak', 'Djavan Ao Vivo - Vol.', 'Djavan Ao Vivo - Vol.', 'Elis Regina-Minha História', 'The Cream Of Clapton', 'Unplugged', 'Album Of The Year', 'Angel Dust', 'King For A Day Fool For A Lifetime', 'The Real Thing', 'Deixa Entrar', 'In Your Honor [Disc ]', 'In Your Honor [Disc ]', 'One By One', 'The Colour And The Shape', 'My Way: The Best Of Frank Sinatra [Disc ]', 'Roda De Funk', 'As Canções de Eu Tu Eles', 'Quanta Gente Veio Ver (Live)', 'Quanta Gente Veio ver--Bônus De Carnaval', 'Faceless', 'American Idiot', 'Appetite for Destruction', 'Use Your Illusion I', 'Use Your Illusion II', 'Blue Moods', 'A Matter of Life and Death', 'A Real Dead One', 'A Real Live One', 'Brave New World', 'Dance Of Death', 'Fear Of The Dark', 'Iron Maiden', 'Killers', 'Live After Death', 'Live At Donington  (Disc )', 'Live At Donington  (Disc )', 'No Prayer For The Dying', 'Piece Of Mind', 'Powerslave', 'Rock In Rio [CD1]', 'Rock In Rio [CD2]', 'Seventh Son of a Seventh Son', 'Somewhere in Time', 'The Number of The Beast', 'The X Factor', 'Virtual XI', 'Sex Machine', 'Emergency On Planet Earth', 'Synkronized', 'The Return Of The Space Cowboy', 'Get Born', 'Are You Experienced?', 'Surfing with the Alien (Remastered)', 'Jorge Ben Jor  Anos', 'Jota Quest-', 'Cafezinho', 'Living After Midnight', 'Unplugged [Live]', 'BBC Sessions [Disc ] [Live]', 'Coda', 'Houses Of The Holy', 'In Through The Out Door', 'IV', 'Led Zeppelin I', 'Led Zeppelin II', 'Led Zeppelin III', 'Physical Graffiti [Disc ]', 'Presence', 'The Song Remains The Same (Disc )', 'The Song Remains The Same (Disc )', 'A TempestadeTempestade Ou O Livro Dos Dias', 'Mais Do Mesmo', 'Greatest Hits', 'Lulu Santos - RCA  Anos De Música - Álbum', 'Lulu Santos - RCA  Anos De Música - Álbum', 'Misplaced Childhood', 'Barulhinho Bom', 'Seek And Shall Find: More Of The Best (-)', 'The Best Of Men At Work', 'Black Album', 'Garage Inc. (Disc )', "Kill 'Em All", 'Load', 'Master Of Puppets', 'ReLoad', 'Ride The Lightning', 'St. Anger', '...And Justice For All', 'Miles Ahead', 'Milton Nascimento Ao Vivo', 'Minas', 'Ace Of Spades', 'Demorou...', 'Motley Crue Greatest Hits', 'From The Muddy Banks Of The Wishkah [Live]', 'Nevermind', 'Compositores', 'Olodum', 'Acústico MTV', 'Arquivo II', 'Arquivo Os Paralamas Do Sucesso', 'Bark at the Moon (Remastered)', 'Blizzard of Ozz', 'Diary of a Madman (Remastered)', 'No More Tears (Remastered)', 'Tribute', 'Walking Into Clarksdale', 'Original Soundtracks', 'The Beast Live', 'Live On Two Legs [Live]', 'Pearl Jam', 'Riot Act', 'Ten', 'Vs.', 'Dark Side Of The Moon', 'Os Cães Ladram Mas A Caravana Não Pára', 'Greatest Hits I', 'News Of The World', 'Out Of Time', 'Green', 'New Adventures In Hi-Fi', 'The Best Of R.E.M.: The IRS Years', 'Cesta Básica', 'Raul Seixas', 'Blood Sugar Sex Magik', 'By The Way', 'Californication', 'Retrospective I (-)', 'Santana - As Years Go By', 'Santana Live', 'Maquinarama', 'O Samba Poconé', 'Judas : B-Sides and Rarities', 'Rotten Apples: Greatest Hits', 'A-Sides', 'Morning Dance', 'In Step', 'Core', 'Mezmerize', '[] Black Light Syndrome', 'Live [Disc ]', 'Live [Disc ]', 'The Singles', 'Beyond Good And Evil', 'Pure Cult: The Best Of The Cult (For Rockers, Ravers, Lovers & Sinners) [UK]', 'The Doors', 'The Police Greatest Hits', 'Hot Rocks, - (Disc )', 'No Security', 'Voodoo Lounge', 'Tangents', 'Transmission', 'My Generation - The Very Best Of The Who', 'Serie Sem Limite (Disc )', 'Serie Sem Limite (Disc )', 'Acústico', 'Volume Dois', 'Battlestar Galactica: The Story So Far', 'Battlestar Galactica, Season', 'Heroes, Season', 'Lost, Season', 'Lost, Season', 'Lost, Season', 'Achtung Baby', "All That You Can't Leave Behind", 'B-Sides -', 'How To Dismantle An Atomic Bomb', 'Pop', 'Rattle And Hum', 'The Best Of -', 'War', 'Zooropa', 'UB40 The Best Of - Volume Two [UK]', 'Diver Down', 'The Best Of Van Halen, Vol. I', 'Van Halen', 'Van Halen III', 'Contraband', 'Vinicius De Moraes', 'Ao Vivo [IMPORT]', 'The Office, Season', 'The Office, Season', 'The Office, Season', 'Un-Led-Ed', 'Battlestar Galactica (Classic), Season', 'Aquaman', 'Instant Karma: The Amnesty International Campaign to Save Darfur', 'Speak of the Devil', '20th Century Masters - The Millennium Collection: The Best of Scorpions', 'House of Pain', 'Radio Brasil (O Som da Jovem Vanguarda) - Seleccao de Henrique Amaro', 'Cake: B-Sides and Rarities', 'LOST, Season', 'Quiet Songs', 'Muso Ko', 'Realize', 'Every Kind of Light', 'Duos II', 'Worlds', 'The Best of Beethoven', 'Temple of the Dog', 'Carry On', 'Revelations', 'Adorate Deum: Gregorian Chant from the Proper of the Mass', 'Allegri: Miserere', 'Pachelbel: Canon & Gigue', 'Vivaldi: The Four Seasons', 'Bach: Violin Concertos', 'Bach: Goldberg Variations', 'Bach: The Cello Suites', 'Handel: The Messiah (Highlights)', 'The World of Classical Favourites', 'Sir Neville Marriner: A Celebration', 'Mozart: Wind Concertos', 'Haydn: Symphonies  -', 'Beethoven: Symhonies Nos.  &', 'A Soprano Inspired', 'Great Opera Choruses', 'Wagner: Favourite Overtures', 'Fauré: Requiem, Ravel: Pavane & Others', 'Tchaikovsky: The Nutcracker', 'The Last Night of the Proms', 'Puccini: Madama Butterfly - Highlights', 'Holst: The Planets, Op.  & Vaughan Williams: Fantasies', "Pavarotti's Opera Made Easy", "Great Performances - Barber's Adagio and Other Romantic Favorites for Strings", 'Carmina Burana', 'A Copland Celebration, Vol. I', 'Bach: Toccata & Fugue in D Minor', 'Prokofiev: Symphony No.', 'Scheherazade', 'Bach: The Brandenburg Concertos', 'Chopin: Piano Concertos Nos.  &', 'Mascagni: Cavalleria Rusticana', 'Sibelius: Finlandia', 'Beethoven Piano Sonatas: Moonlight & Pastorale', 'Great Recordings of the Century - Mahler: Das Lied von der Erde', 'Elgar: Cello Concerto & Vaughan Williams: Fantasias', 'Adams, John: The Chairman Dances', "Tchaikovsky:  Festival Overture, Op., Capriccio Italien & Beethoven: Wellington's Victory", 'Palestrina: Missa Papae Marcelli & Allegri: Miserere', 'Prokofiev: Romeo & Juliet', 'Strauss: Waltzes', 'Berlioz: Symphonie Fantastique', 'Bizet: Carmen Highlights', 'English Renaissance', 'Handel: Music for the Royal Fireworks (Original Version )', 'Grieg: Peer Gynt Suites & Sibelius: Pelléas et Mélisande', 'Mozart Gala: Famous Arias', 'SCRIABIN: Vers la flamme', 'Armada: Music from the Courts of England and Spain', 'Mozart: Symphonies Nos.  &', 'Back to Black', 'Frank', 'Carried to Dust (Bonus Track Version)', "Beethoven: Symphony No.  'Pastoral' Etc.", 'Bartok: Violin & Viola Concertos', "Mendelssohn: A Midsummer Night's Dream", 'Bach: Orchestral Suites Nos.  -', 'Charpentier: Divertissements, Airs & Concerts', 'South American Getaway', 'Górecki: Symphony No.', 'Purcell: The Fairy Queen', 'The Ultimate Relexation Album', 'Purcell: Music for the Queen Mary', 'Weill: The Seven Deadly Sins', 'J.S. Bach: Chaconne, Suite in E Minor, Partita in E Major & Prelude, Fugue and Allegro', 'Prokofiev: Symphony No. & Stravinksy: Le Sacre Du Printemps', 'Szymanowski: Piano Works, Vol.', 'Nielsen: The Six Symphonies', "Great Recordings of the Century: Paganini's  Caprices", "Liszt -  Études D'Execution Transcendante", 'Great Recordings of the Century - Shubert: Schwanengesang,  Lieder', 'Locatelli: Concertos for Violin, Strings and Continuo, Vol.', 'Respighi:Pines of Rome', "Schubert: The Late String Quartets & String Quintet ( CD's)", "Monteverdi: L'Orfeo", 'Mozart: Chamber Music', 'Koyaanisqatsi (Soundtrack from the Motion Picture)']
"""

texts = artists + albums

embeddings = OpenAIEmbeddings(
    api_key=os.environ['OPEN_API_KEY'],
)
vector_db = FAISS.from_texts(texts, embeddings) # 벡터 저장소 저장
retriever = vector_db.as_retriever() # retriever를 통해 검색기 생성

retriever_tool = create_retriever_tool(
    retriever,
    name="name_search",
    description="이름, 성 주소 등 데이터가 실제로 어떻게 쓰여졌는지 알아내는 데 사용합니다.",
)

custom_tool_list = [retriever_tool]

#

llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0,
    api_key=os.environ['OPEN_API_KEY'],
)

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=llm
)

custom_suffix = """
사용자가 고유명사를 기준으로 필터링해 달라고 요청하는 경우, 먼저 name_search 도구를 사용하여 철자를 확인해야 합니다.
그렇지 않으면 데이터베이스의 테이블을 살펴보고 쿼리할 수 있는 항목을 확인할 수 있습니다.
그런 다음 가장 관련성이 높은 테이블의 스키마를 쿼리해야 합니다.
"""

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    extra_tools=custom_tool_list,
    suffix=custom_suffix,
)

agent.run("elice in chains는 몇 개의 앨범을 가지고 있나요?")
"""
> Entering new SQL Agent Executor chain...

Invoking: `name_search` with `{'query': 'elice in chains'}`


Elis Regina

Alice In Chains

Powerslave

Out Of Exile사용자가 "Alice In Chains"를 의미했을 가능성이 높습니다. 
이제 이 정보를 사용하여 데이터베이스에서 앨범 수를 찾을 수 있습니다. 
먼저, 데이터베이스에 어떤 테이블이 있는지 확인해야 합니다.

> Finished chain.
"""

agent.run("alice in chains는 몇 개의 앨범을 가지고 있나요?")
"""
> Entering new SQL Agent Executor chain...

Invoking: `name_search` with `{'query': 'alice in chains'}`


Alice In Chains

Metallica

Pearl Jam

Pearl Jam
Invoking: `sql_db_list_tables` with `{}`


albums, artists, customers, employees, genres, invoice_items, invoices, media_types, playlist_track, playlists, tracks
Invoking: `sql_db_schema` with `{'table_names': 'albums, artists'}`



CREATE TABLE albums (
	"AlbumId" INTEGER NOT NULL, 
	"Title" NVARCHAR(160) NOT NULL, 
	"ArtistId" INTEGER NOT NULL, 
	PRIMARY KEY ("AlbumId"), 
	FOREIGN KEY("ArtistId") REFERENCES artists ("ArtistId")
)

/*
3 rows from albums table:
AlbumId	Title	ArtistId
1	For Those About To Rock We Salute You	1
2	Balls to the Wall	2
3	Restless and Wild	2
*/


CREATE TABLE artists (
	"ArtistId" INTEGER NOT NULL, 
	"Name" NVARCHAR(120), 
	PRIMARY KEY ("ArtistId")
)

/*
3 rows from artists table:
ArtistId	Name
1	AC/DC
2	Accept
3	Aerosmith
*/
Invoking: `sql_db_query_checker` with `{'query': "SELECT COUNT(*) FROM albums WHERE ArtistId = (SELECT ArtistId FROM artists WHERE Name = 'Alice In Chains')"}`


SELECT COUNT(*) FROM albums WHERE ArtistId = (SELECT ArtistId FROM artists WHERE Name = 'Alice In Chains')
Invoking: `sql_db_query` with `{'query': "SELECT COUNT(*) FROM albums WHERE ArtistId = (SELECT ArtistId FROM artists WHERE Name = 'Alice In Chains')"}`


[(1,)]Alice In Chains는 1개의 앨범을 가지고 있습니다.

> Finished chain.
"""
