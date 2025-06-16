from dotenv import load_dotenv
load_dotenv()

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRelevancyMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

# 3 test cases corresponding to the 3 files uploaded to Qdrant
# (You can update these with your own test cases as needed)
test_cases = [
    LLMTestCase(
        input="What is the Tush Push play in the NFl?",
        actual_output="The Tush Push Play in the NFL is a running play in which the ball is snapped to the quarterback, who then plunges forward into the offensive line. While the linemen push forward, the quarterback is then pushed from behind by a tight end and a running back, akin to a scrum in rugby.",
        expected_output="The Tush Push is a quarterback sneak variation popularized by the Eagles, where players push the quarterback forward to gain short yardage. It has been successful but controversial due to safety and competitiveness concerns.",
        context=[
            "The 'Tush Push' is a quarterback sneak tactic where teammates push the QB forward after the snap.",
            "Popularized by the Eagles in 2022–2023 seasons.",
            "Often used on 3rd/4th and 1-yard situations.",
            "Has sparked NFL debate about player safety and potential rule changes."
        ],
        retrieval_context=[
            """It is a running play in which the ball is snapped to the quarterback, who then plunges forward into the offensive line. While the linemen push forward, the quarterback is then 
pushed from behind by a tight end and a running back, akin to a scrum in rugby.
Related article
The ‘Brotherly Shove’: Why in vogue ‘Tush Push’ has become unstoppable play in the NFL

[Score 0.63] 
It is a running play in which the ball is snapped to the quarterback, who then plunges forward into the offensive line. While the linemen push forward, the quarterback is then 
pushed from behind by a tight end and a running back, akin to a scrum in rugby.
Related article
The ‘Brotherly Shove’: Why in vogue ‘Tush Push’ has become unstoppable play in the NFL

[Score 0.63] 
It is a running play in which the ball is snapped to the quarterback, who then plunges forward into the offensive line. While the linemen push forward, the quarterback is then 
pushed from behind by a tight end and a running back, akin to a scrum in rugby.
Related article
The ‘Brotherly Shove’: Why in vogue ‘Tush Push’ has become unstoppable play in the NFL

[Score 0.56] The combined efforts usually result in a short-yardage gain that is enough for either a first down or a touchdown and the Eagles’ version of it is usually 
unstoppable.
Eagles quarterback Jalen Hurts – who is the person with the ball in his hands and is being pushed from behind – has benefitted greatly from this play, with the majority of his 52 
rushing touchdowns over the last four seasons coming from the tush push.
It became a key driving factor in the team reaching the Super Bowl two years ago and in their title success last season.
Like many other aspects across the NFL, other teams have tried to adopt the tush push with varying success, while the Eagles remain the masters of it.
Why do teams want it outlawed?
Despite the success of the tush push, it has become a controversial play, with some arguing it takes away competitiveness and makes football less exciting.
The play, which bares similarities to the old-school quarterback sneak used in the early days of football, has also led to safety concerns, with players pushing against one 
another with all their force in such close proximity.
Green Bay, which was beaten handily by the Eagles in the wild card round of the playoffs as Philadelphia went on to win Super Bowl LIX, was the team to table the motion to ban the
play, with CEO and team president Mark Murphy saying the tush push was “bad for the game.”
“There is no skill involved and it is almost an automatic first down on plays of a yard or less,” Murphy added. “We should go back to prohibiting the push of the runner. This 
would bring back the traditional QB sneak.
[Score 0.56] The combined efforts usually result in a short-yardage gain that is enough for either a first down or a touchdown and the Eagles’ version of it is usually 
unstoppable.
Eagles quarterback Jalen Hurts – who is the person with the ball in his hands and is being pushed from behind – has benefitted greatly from this play, with the majority of his 52 
rushing touchdowns over the last four seasons coming from the tush push.
It became a key driving factor in the team reaching the Super Bowl two years ago and in their title success last season.
Like many other aspects across the NFL, other teams have tried to adopt the tush push with varying success, while the Eagles remain the masters of it.
Why do teams want it outlawed?
Despite the success of the tush push, it has become a controversial play, with some arguing it takes away competitiveness and makes football less exciting.
The play, which bares similarities to the old-school quarterback sneak used in the early days of football, has also led to safety concerns, with players pushing against one 
another with all their force in such close proximity.
Green Bay, which was beaten handily by the Eagles in the wild card round of the playoffs as Philadelphia went on to win Super Bowl LIX, was the team to table the motion to ban the
play, with CEO and team president Mark Murphy saying the tush push was “bad for the game.”
“There is no skill involved and it is almost an automatic first down on plays of a yard or less,” Murphy added. “We should go back to prohibiting the push of the runner. This 
would bring back the traditional QB sneak."""
        ]
    ),
    LLMTestCase(
        input="Did Max verstappen win the 2025 Japan Grand Prix?",
        actual_output="Max Verstappen won the 2025 Japan Grand Prix.",
        expected_output="Max Verstappen won the 2025 Japan Grand Prix, dominating the race weekend and securing first place at Suzuka.",
        context=[
            "Max Verstappen won the 2025 Japanese Grand Prix at Suzuka.",
            "He secured pole position and led for most of the race.",
            "Red Bull's performance was praised despite recent car struggles.",
            "The win put Verstappen just one point behind Lando Norris in the drivers' standings.",
            "The race was seen as one of Verstappen's best performances.",
            "McLaren drivers Norris and Piastri finished behind him.",
            "Christian Horner and Alonso praised Verstappen's skill."
        ],
        retrieval_context=[
            """[Score 0.51]  Bar a short period in the pit-stop phase and a close shave with Lando Norris on pit exit, Verstappen had controlled proceedings out in front. Red Bull boss Christian
Horner labeled the four-time world champion's performance "inspirational" in his own congratulatory message on the cool-down lap.
The victory moved Verstappen a point behind Norris in the championship after three races, a remarkable feat given both McLaren's pace so far this season and Red Bull's ongoing 
struggles with its RB21 car. It was a weekend that continued to bolster his growing legend as a man able to do special things with whatever machine he has to drive.
Never in doubt
The lead-in to the Japanese Grand Prix had centered on Red Bull's decision to hand Verstappen his third different teammate in only four races, with Yuki Tsunoda stepping in to 
replace Liam Lawson. The latter had become the latest victim of the poisoned chalice that is the second Red Bull seat, finishing last in both the sprint and the grand prix in 
China. Tsunoda actually started the weekend strongly in terms of pace, finishing close to Verstappen in both of Friday's practice sessions, but a scruffy lap in Q2 saw him qualify
down the order when it mattered.
The incredible Verstappen pole lap that followed in Q3 highlighted the contrast between the new teammates. Two-time world champion and Le Mans 24 Hour winner Fernando Alonso, one 
of the most complete racing talents of the modern era, watched in awe in the media pen as the lap unfolded and the Dutchman's name moved to the top of the timing screens.
"He's an outstanding driver. He's proving it every weekend," the Aston Martin driver said afterward.
[Score 0.49] "
Max Verstappen beat McLaren's Lando Norris and Oscar Piastri at Suzuka to go one point off the top of the drivers' championship. Clive Mason/Getty Images
That rising reference point has ultimately been the crux of Red Bull's issues with the second car.
[Score 0.47] Vertsappen: Incredibly happy with Japan GP winRed Bull's Max Verstappen along with Lando Norris, Oscar Piastri and Charles Leclerc react to the Japan GP.
Whether Verstappen can stay in the fight now might be the biggest story of this portion of the season, especially if Red Bull can do with their car what McLaren was able to 
achieve with its own 12 months ago. Asked on Sunday whether he felt Verstappen could stay in the fight all year, Alonso had memories of his own 2012 season when he infamously came
agonizingly short of a third world championship in a car considered to be second, third and fourth quickest at different parts of the season.
Shaking his head on hearing the question, Alonso said: "I don't know, I've been there, I've been fighting for world championships with the fourth-fastest car, so you know, at the 
end it's difficult, as it was difficult for me to beat [Sebastian] Vettel. I hope for him he can fight until the end, but they need to improve a little bit the machine.
"He won four, so he can fight with a little bit [of a] less competitive car, but yeah, it's hard and I think the people don't realize how difficult it is and how you need to make 
it perfect every weekend, and he's doing it so far. It reminds me of my 2012 in a way, when the car was not so good and we fought for the championship, but to win it at the end, 
let's see, I hope for him he can win it."
McLaren's fight will be a boon to Verstappen, too. Piastri has had an impressive start to the season, and his pace has been much closer to Norris (and on a more consistent basis) 
than it was last year. The fact the Australian was hinting at a position swap with Norris in the closing stages gives a tantalizing glimpse at the headaches McLaren has been 
anticipating from its two mega-talented race drivers as this season progresses.
As has always been the case when teammates have fought together for a championship, a flashpoint will be a matter of when -- not if -- especially in a season spanning 24 races. 
Any driver at a rival team would benefit from that situation - for a driver as talented as Red Bull's lead, it could be the perfect opportunity to secure a legitimate shot at a 
fifth title, regardless of how much the team improves the car.
Verstappen responded to the suggestion of his championship challenge in typical fashion. When asked if 2025 is a three-way fight, he said: "I don't know. I mean, there's still ...
it's a very long season."""
        ]
    ),
    LLMTestCase(
        input="Did Duke win their 2025 March Madness matchup against Houston Men's basketball?",
        actual_output="No, Duke did not win their 2025 March Madness matchup against Houston Men's basketball. According to the provided context, Duke lost to Houston in the Final Four.",
        expected_output="No, Duke lost to Houston in the 2025 Final Four. Houston scored the last nine points to win 70-67.",
        context=[
            "Duke had a six-point lead in the final 35 seconds of the game.",
            "Houston scored the final nine points to win 70-67.",
            "Cooper Flagg missed a potential game-winning shot in the closing seconds.",
            "Flagg committed a foul that allowed Houston to take the lead.",
            "The game ended in a heartbreaking loss for Duke in the Final Four."
        ],
        retrieval_context=[
                """    [Score 0.33] Houston's defenders were their marauding selves all night, with the most jarring statistic in the box score being that of Duke center Khaman Maluach when he 
failed to grab a rebound in more than 21 minutes of play and ending the night with a plus-minus of -20.
Roberts' final salvo was getting a tough contest on Flagg's potential game winner.
"I thought he did an awesome job of getting his hands up high enough that it wasn't an easy look," Sampson said of Roberts. "Some tough shots all night."
Flagg finished the contest with 27 points, shooting 8-for-19 from the field. He got little help, as Duke had only one field goal over the game's last 10:30.
He rode back to the Duke locker room in a golf cart at 11:54 p.m., staring into space with a towel wrapped around his neck. Flagg entered the cone of silence suddenly facing the 
end of a season and likely a college career.
Three minutes later, Duke coach Jon Scheyer rode past with his wife next to him and athletic director Nina King sitting in the back. After leading by as much as 14, Duke had just 
coughed up the fifth-biggest lead in Final Four history. The loss will echo, just like that slamming door, long into the offseason.
"I keep going back, we're up six with under a minute to go," Scheyer said.
"We just have to finish the deal."
[Score 0.33] Houston's defenders were their marauding selves all night, with the most jarring statistic in the box score being that of Duke center Khaman Maluach when he failed to
grab a rebound in more than 21 minutes of play and ending the night with a plus-minus of -20.
Roberts' final salvo was getting a tough contest on Flagg's potential game winner.
"I thought he did an awesome job of getting his hands up high enough that it wasn't an easy look," Sampson said of Roberts. "Some tough shots all night."
Flagg finished the contest with 27 points, shooting 8-for-19 from the field. He got little help, as Duke had only one field goal over the game's last 10:30.
He rode back to the Duke locker room in a golf cart at 11:54 p.m., staring into space with a towel wrapped around his neck. Flagg entered the cone of silence suddenly facing the 
end of a season and likely a college career.
Three minutes later, Duke coach Jon Scheyer rode past with his wife next to him and athletic director Nina King sitting in the back. After leading by as much as 14, Duke had just 
coughed up the fifth-biggest lead in Final Four history. The loss will echo, just like that slamming door, long into the offseason.
"I keep going back, we're up six with under a minute to go," Scheyer said.
"We just have to finish the deal."
[Score 0.32] For Duke, stunned silence after epic collapse in Final Four
play
Pete Thamel
Apr 6, 2025, 03:27 AM ET

[Score 0.32] For Duke, stunned silence after epic collapse in Final Four
play
Pete Thamel
Apr 6, 2025, 03:27 AM ET

[Score 0.31] Verstappen's perfect Japan GP shows McLaren won't get easy ride
play
Nate Saunders
Apr 6, 2025, 09:00 AM ET


    Graph Search Results:
    Duke —[trailing by]-> one point (source: duke_article.txt)
Duke —[led by]-> one point (source: duke_article.txt)
Duke —[participated in]-> Final Four (source: duke_article.txt)
Duke —[part of]-> Final Four (source: duke_article.txt)
Duke —[lost to]-> Houston (source: duke_article.txt)
Duke —[had]-> six-point lead (source: duke_article.txt)
Duke —[has team]-> Blue Devils (source: duke_article.txt)
Duke —[led by]-> 14 (source: duke_article.txt)"""
        ]
    )
]

# Define evaluation metrics
metrics = [
    AnswerRelevancyMetric(model="gpt-4o-mini", threshold=0.7),
    FaithfulnessMetric(model="gpt-4o-mini", threshold=0.7),
    ContextualPrecisionMetric(model="gpt-4o-mini", threshold=0.7),
    ContextualRelevancyMetric(model="gpt-4o-mini", threshold=0.7),
    ContextualRecallMetric(model="gpt-4o-mini", threshold=0.7)
]

# Run the tests
for test_case in test_cases:
    # Manually evaluate
    for metric in metrics:
        metric.measure(test_case)
        print(f"\n {metric.__class__.__name__}")
        print(f"Score: {metric.score}")
        if hasattr(metric, "reason"):
            print(f"Reason: {metric.reason}")
