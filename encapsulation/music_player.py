class MusicPlayer:
    def __init__(self, song, playing, volume):
        self._song = song
        self._playing = playing
        self._volume = volume

    @property
    def song(self):
        return self._song
    
    @property
    def playing(self):
        return self._playing
    
    @property
    def volume(self):
        return self._volume
    
    @volume.setter
    def volume(self, value):
        value = int(value)
        if value < 0:
            raise ValueError('The value cannot be less than zero') 
        if value > 100:
            raise ValueError('Volume cannot be greater than 100')
        self._volume = min(value, 100) 

    def play(self):
        self._playing = True
        if self._volume <= 0:
            print(f'You wont hear the song and volume cannot be negative')
            return
        state = 'Playing' if self._playing else 'Paused'
        print(f'{state}: {self._song} at volume {self.volume}')

    def pause(self):
        self._playing = False
        print(f'Paused')

    def change_song(self, new_song):
        self._song = new_song
        print(f'Now playing: {new_song}')

    def volume_up(self):
        try:
            self.volume += 10
            print(f"Volume up → {self.volume}")
        except ValueError as e:
            print(e)

    def volume_down(self):
        try:
            self.volume -= 10
            print(f"Volume down → {self.volume}")
        except ValueError as e:
            print(e)

    def __str__(self):
        return f'Music Player | Song: {self.song} | Volume: {self.volume}'


# Test
if __name__ == "__main__":
    player = MusicPlayer('Essence - Wizkid', 'playing', 40)
    print(player)
    
    player.play()
    print(player)
    
    player.volume_up()
    player.volume_up()
    print(player)
    
    for _ in range(10):
        player.volume_down()
    print(player)
    
    player.change_song("Ojuelegba - Wizkid")
    player.play()
    print(player)
    
    player.pause()
    print(player)