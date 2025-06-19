from rag_system import RAGSystem

def demo_with_sample_text():
    """Demo the RAG system with sample board game text"""
    
    # Initialize the system
    print("🚀 Initializing RAG System...")
    rag = RAGSystem()
    
    # Sample extracted text (replace with your actual extracted text)
    board_game_text = """
    Monopoly Board Game Rules
    
    Monopoly is a real estate board game for 2 to 8 players. The goal is to become the wealthiest player 
    through buying, renting, and selling properties. Players take turns moving around the board by rolling 
    two dice. The board has 40 spaces including 28 properties, 4 railroad stations, 2 utilities, 
    3 Chance spaces, 3 Community Chest spaces, a Luxury Tax space, an Income Tax space, and the four 
    corner squares: GO, Jail, Free Parking, and Go to Jail.
    
    Setup: Each player chooses a token and places it on GO. Each player receives $1,500 in Monopoly money. 
    The banker distributes the money: two $500s, two $100s, two $50s, six $20s, five $10s, five $5s, 
    and five $1s to each player.
    
    Gameplay: Players take turns in clockwise order. On each turn, a player rolls the dice and moves 
    their token clockwise around the board the number of spaces indicated by the dice. After moving, 
    the player performs actions based on the space they landed on.
    
    Buying Properties: When a player lands on an unowned property, they may buy it from the bank at 
    the listed price. If they choose not to buy it, the property is auctioned to the highest bidder.
    
    Collecting Rent: When a player lands on a property owned by another player, they must pay rent 
    to the property owner. The amount of rent depends on the property and whether the owner has 
    developed it with houses or hotels.
    
    Building Houses and Hotels: Players can build houses and hotels on their properties to increase 
    rent. A player must own all properties in a color group before building. Houses must be built 
    evenly across all properties in the group.
    
    Special Spaces: Landing on Chance or Community Chest spaces requires drawing a card and following 
    its instructions. Landing on tax spaces requires paying the specified amount to the bank.
    
    Winning: The game continues until all but one player have gone bankrupt. The last remaining player 
    is the winner.
    
    CodeNames Rules
    
    Codenames is a word-based board game for 4-8 players divided into two teams. Each team has a 
    spymaster who gives one-word clues to help their teammates identify their team's agents among 
    25 word cards laid out in a 5x5 grid.
    
    Setup: Place 25 word cards in a 5x5 grid. The spymasters sit across from each other, and their 
    teammates sit across from each other. Place the key card in the card stand so only spymasters 
    can see it. This card shows which words belong to which team.
    
    Teams: There are red agents, blue agents, innocent bystanders, and one assassin. One team has 
    9 agents, the other has 8. The team with 9 agents goes first.
    
    Giving Clues: The spymaster gives a one-word clue followed by a number. The clue relates to the 
    meaning of the words their team should guess. The number tells the team how many words relate 
    to the clue.
    
    Guessing: The team discusses and makes guesses. They can guess up to one more than the number 
    given with the clue. If they guess correctly, they place their team's agent card on that word. 
    If they guess wrong, their turn ends.
    
    Winning: A team wins by contacting all their agents first. A team also wins immediately if the 
    other team contacts the assassin.
    """
    
    # Add the text to the RAG system
    print("📄 Adding text to the system...")
    rag.add_text(board_game_text, "board_game_rules.txt")
    
    # Show statistics
    stats = rag.get_stats()
    print(f"✅ System ready! Database contains {stats['total_chunks']} chunks\n")
    
    # Demo queries
    demo_questions = [
        "How do you win in Monopoly?",
        "What is the setup for CodeNames?",
        "How much money does each player start with in Monopoly?",
        "How many players can play CodeNames?",
        "What happens when you land on someone else's property?",
        "How do spymasters give clues in CodeNames?"
    ]
    
    print("🎯 Demo Questions & Answers:")
    print("=" * 60)
    
    for question in demo_questions:
        print(f"\n❓ Q: {question}")
        result = rag.query(question)
        print(f"💡 A: {result['answer']}")
        print(f"📚 Source: {', '.join(result['sources'])}")
        
        # Show confidence score
        if result['similarity_scores']:
            avg_score = sum(result['similarity_scores']) / len(result['similarity_scores'])
            confidence = (1 - avg_score) * 100
            print(f"🎯 Confidence: {confidence:.1f}%")
        print("-" * 40)
    
    return rag

def interactive_mode(rag):
    """Start interactive question-answering mode"""
    print("\n" + "=" * 60)
    print("🤖 Interactive Mode - Ask your own questions!")
    print("Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        question = input("\n❓ Your question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if question:
            print("🔍 Searching...")
            result = rag.query(question)
            print(f"\n💡 Answer: {result['answer']}")
            print(f"📚 Sources: {', '.join(result['sources'])}")

def main():
    """Main function to run the demo"""
    print("🎲 Board Game RAG System Demo")
    print("=" * 40)
    
    # Run demo with sample text
    rag = demo_with_sample_text()
    
    # Ask user if they want interactive mode
    while True:
        choice = input("\n🤔 Would you like to ask your own questions? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            interactive_mode(rag)
            break
        elif choice in ['n', 'no']:
            print("👋 Thanks for trying the demo!")
            break
        else:
            print("Please enter 'y' or 'n'")

if __name__ == "__main__":
    main()