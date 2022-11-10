import pandas as pd
import json

class input_processor:
    def __init__(self):
        self.skills_list = ['Very Low Skill', 'Low Skill', 'Normal Skill', 'High Skill', 'Very High Skill']
        self.heroes_list = ['Unknown', 'Anti-Mage', 'Axe', 'Bane', 'Bloodseeker', 'Crystal Maiden', 'Drow Ranger',
                            'Earthshaker', 'Juggernaut', 'Mirana', 'Morphling', 'Shadow Fiend',
                            'Phantom Lancer', 'Puck', 'Pudge', 'Razor', 'Sand King', 'Storm Spirit', 'Sven',
                            'Tiny', 'Vengeful Spirit', 'Windranger', 'Zeus', 'Kunkka', 'Lina', 'Lion',
                            'Shadow Shaman', 'Slardar', 'Tidehunter', 'Witch Doctor', 'Lich', 'Riki',
                            'Enigma', 'Tinker', 'Sniper', 'Necrophos', 'Warlock', 'Beastmaster',
                            'Queen of Pain', 'Venomancer', 'Faceless Void', 'Wraith King',
                            'Death Prophet', 'Phantom Assassin', 'Pugna', 'Templar Assassin', 'Viper',
                            'Luna', 'Dragon Knight', 'Dazzle', 'Clockwerk', 'Leshrac', "Nature's Prophet",
                            'Lifestealer', 'Dark Seer', 'Clinkz', 'Omniknight', 'Enchantress', 'Huskar',
                            'Night Stalker', 'Broodmother', 'Bounty Hunter', 'Weaver', 'Jakiro',
                            'Batrider', 'Chen', 'Spectre', 'Ancient Apparition', 'Doom', 'Ursa',
                            'Spirit Breaker', 'Gyrocopter', 'Alchemist', 'Invoker', 'Silencer',
                            'Outworld Devourer', 'Lycan', 'Brewmaster', 'Shadow Demon', 'Lone Druid',
                            'Chaos Knight', 'Meepo', 'Treant Protector', 'Ogre Magi', 'Undying', 'Rubick',
                            'Disruptor', 'Nyx Assassin', 'Naga Siren', 'Keeper of the Light', 'Io',
                            'Visage', 'Slark', 'Medusa', 'Troll Warlord', 'Centaur Warrunner', 'Magnus',
                            'Timbersaw', 'Bristleback', 'Tusk', 'Skywrath Mage', 'Abaddon', 'Elder Titan',
                            'Legion Commander', 'Techies', 'Ember Spirit', 'Earth Spirit', 'Underlord',
                            'Terrorblade', 'Phoenix', 'Oracle', 'Winter Wyvern', 'Arc Warden']


    def process_input(self, json_input):
        region = json_input["region"]
        cluster = json_input["cluster"]
        day_of_week = json_input["day_of_week"]
        time = json_input["time"]
        radiant_winrate = json_input["radiant_winrate"]
        dire_winrate = json_input["dire_winrate"]
        radiant_skills = ['radiant_' + item for item in json_input["radiant_skills"]]
        dire_skills = ['dire_' + item for item in json_input["dire_skills"]]
        radiant_heroes= ['radiant_' + item for item in json_input["radiant_heroes"]]
        dire_heroes = ['dire_' + item for item in json_input["dire_heroes"]]

        # Skills and Heroes
        # Restructuring to replicate the schema of the desired dataframe for modelling
        radiant_skills_cols = list(map(lambda s: 'radiant_' + s, list(map(lambda x: x.lower(), self.skills_list))))
        dire_skills_cols = list(map(lambda s: 'dire_' + s, list(map(lambda x: x.lower(), self.skills_list))))

        radiant_hero_cols = list(map(lambda s: 'radiant_' + s, list(map(lambda x: x.lower(), self.heroes_list))))
        dire_hero_cols = list(map(lambda s: 'dire_' + s, list(map(lambda x: x.lower(), self.heroes_list))))

        # Skills
        radiant_skills_dict = {}
        dire_skills_dict = {}

        for item in radiant_skills_cols:
            count_list = []
            for l in radiant_skills:
                count_list.append(l.count(item))
                
            radiant_skills_dict[item] = count_list

        for item in dire_skills_cols:
            count_list = []
            for l in dire_skills:
                count_list.append(l.count(item))
                
            dire_skills_dict[item] = count_list

        radiant_skills_new = [sum(values) for key, values in radiant_skills_dict.items()]
        dire_skills_new = [sum(values) for key, values in dire_skills_dict.items()]

        # Heroes
        radiant_heroes_dict = {}
        dire_heroes_dict = {}

        for item in radiant_hero_cols:
            count_list = []
            for l in radiant_heroes:
                count_list.append(l.count(item))
                
            radiant_heroes_dict[item] = count_list

        for item in dire_hero_cols:
            count_list = []
            for l in dire_heroes:
                count_list.append(l.count(item))
                
            dire_heroes_dict[item] = count_list

        radiant_heroes_new = [sum(values) for key, values in radiant_heroes_dict.items()]
        dire_heroes_new = [sum(values) for key, values in dire_heroes_dict.items()]

        ##
        output_data = [region, cluster, day_of_week, time,
                       radiant_winrate, dire_winrate]
        output_data.extend(radiant_skills_new + dire_skills_new + radiant_heroes_new + dire_heroes_new)
        
        print(f'processed_data {type(output_data)} \n {output_data} \n ')

        columns = ['region','cluster','day_of_week','time','radiant_winrate','dire_winrate']
        columns.extend(radiant_skills_cols + dire_skills_cols + radiant_hero_cols + dire_hero_cols)

        df = pd.DataFrame([output_data], columns = columns)

        output = df.to_dict(orient='records')

        return output

    